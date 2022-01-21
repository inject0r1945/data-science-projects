import os.path
import os
import random
from copy import copy

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from IPython import display
import pandas as pd
import numpy as np
import torchvision.transforms as T
from torchvision.io import read_image
from torchvision import models
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import timm
import tqdm

from PIL import Image
import imagehash
import imageio
from tqdm import tqdm


def set_seed(seed=None, seed_torch=True):
    """ Фиксируем генератор случайных чисел
    
    Параметры
    ---------
    seed : int
    seed_torch : bool
      Если True, то будет зафиксирован PyTorch
    """
    
    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class TransferNet:
    """ Создание модели для трансферного обучения

    В качестве экстратктора признаков берется сверточная сеть из VGG16, ResNet18, GoogleNet и AlexNet.
    В экстракторе признаков замораживаются градиенты, кроме небольшой части сверточных блоков в конце экстрактора.
    И кроме слоев Batch Normalization. К выходу экстрактора признаков добавляется полносвязный слой.
    Параметры полносвязного слоя регулируются параметрами класса.

    Параметры
    ---------
    output_dims : list
      Структура полносвязной сети на выходе экстратора признаков.
      Длина списка - количество полносвязных слоев.
      Значения списка - кол-во нейронов в каждом полносвязном слое.
      Пример: [1024, 512, 256, 1] - четыре полносвязных слоя.
              Первый слой с 1024 нейронами, второй слой с 512 нейронами,
              третий слой с 256 нейронами и последний слой с 1 нейроном.
      По умолчанию [128, 256, 1].
    pretrained : bool
      Если True, то будут загружена предобученная модель на ImageNet.
      По умолчанию True.
      !!! ВАЖНО: Класс не тестировался на не предобученных моделях.
          Структура модели при значении pretrained=False может отличаться, поведение класса может быть некорректным.
    full_trainable : bool
      Сделать ли модель полностью обучаемой. По умолчанию False. Если False, то для обучения доступна только
      полносвязная сеть и около 10% окончания экстрактора признаков
    seed : int
      Фиксация генератора случайных чисел.
    dropout : float
      Значение вероятности для всех слоев Dropout полносвязной сети.
      По умолчанию 0.5
    """

    def __init__(self, output_dims=None, dropout=0.5, pretrained=True, full_trainable=False, seed=None):

        if seed is not None:
            set_seed(seed)

        if not output_dims:
            output_dims = [128, 256, 1]

        output_dims =list(output_dims)

        assert 0.0 <= dropout <= 1.0, f"Значение dropout может находитсья в диапазоне от 0 до 1. Текущее значение: ({dropout})"
        assert isinstance(pretrained, bool), f"Значение pretrained должно иметь тип bool. Текущее значение: ({pretrained})"
        assert len(output_dims) > 0, f"Список output_dims должен быть больше 0. Текущее значение: ({output_dims})"
        assert all([isinstance(x, int) for x in output_dims]), f"Все значения в output_dims должны быть целыми. Текущее значение: ({output_dims})"

        self.output_dims = output_dims
        self.dropout = dropout
        self.pretrained = pretrained
        self.full_trainable = full_trainable

    def _make_fc(self, input_dim, first_dropout=True):
        """" Создание полносвязного слоя

        Количество полносвязных слоев и количество нейронов берется из переменной self.output_dims
        Значение вероятности Dropout берется из self.dropout

        Параметры
        ---------
        input_dim : int
          Размер входа полносвязной сети
        first_dropout : bool
          Если True, то первый слой полносвязной сети - Dropout. Иначе Linear

        Результат
        ---------
        fc : torch Sequential
          Последовательность слоев типа nn.Linear
        """

        if first_dropout:
            layers = [nn.Dropout(self.dropout)]
        else:
            layers = []

        # На основе self.output_dims конфигурируем каждый полносвязный слой
        for index, output_dim in enumerate(self.output_dims):
            layers.append(nn.Linear(input_dim, output_dim))
            # Если слой не последний, то после него добавляем Relu и Dropout
            if index != len(self.output_dims) - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
            input_dim = output_dim

        return nn.Sequential(*layers)

    def _prepare_transfernet(self, transfer_model, no_grad_layer=None, first_dropout=True):
        """ Подготовка сети для трансферного обучения

        Параметры
        ---------
        transfer_model : torch model
          Модель из torchvision.models
        no_grad_layer : str
          Название слоя, вплоть до которого будет выключено обучение параметров. По умолчанию None.
          Если None, то все параметры модели будут обучаться.
        first_dropout : bool
          Если True, то первый слой полносвязной сети - Dropout. Иначе Linear

        Результат
        ---------

        """

        if no_grad_layer is not None:

            # Замораживаем градиенты у модели, отключаем обучение параметров
            for name, param in transfer_model.named_parameters():

                # Выключаем градиенты во всех блоках модели вплоть до no_grad_layer
                if no_grad_layer in name:
                    break
                # Не выключаем обучение параметров у всех блоков Batch Normalization
                if 'bn' in name:
                    continue

                param.requires_grad = False

        # Находим размер выхода экстрактора признаков модели
        # Значение находится в свойстве in_features у первого слоя типа Linear после экстрактора
        # Модели не однотипные, полносвязный слой может храниться в параметре fc или classifier
        if 'fc' in transfer_model.__dir__():
            fc_name = 'fc'
            fc_in_features = transfer_model.fc.in_features
        elif 'head' in transfer_model.__dir__():
            fc_name = 'head'
            fc_in_features = transfer_model.head.in_features
        elif 'classifier' in transfer_model.__dir__():
            fc_name = 'classifier'
            if isinstance(transfer_model.classifier, nn.Linear):
                fc_in_features = transfer_model.classifier.in_features
            else:
                # Обычно модели с блоком classifier содержат последовательность слоев.
                # Бывает такое, что первым слоем идет Dropout и у него нет параметра in_features
                try:
                    fc_in_features = transfer_model.classifier[0].in_features
                except:
                    fc_in_features = transfer_model.classifier[1].in_features
        else:
            raise Exception("В модели не найден блок полносвязной сети")

        # Заменяем у модели полносвязный слой на сгенерированный
        setattr(transfer_model, fc_name, self._make_fc(input_dim=fc_in_features, first_dropout=first_dropout))

        return transfer_model

    def _make_swin_base_patch4_window7_224(self):
        """ Создание модели на базе swin_base_patch4_window7_224

        Используетя параметр self.pretrained для загрузки предобученной модели.

        Результат
        ---------
        model : torch model
          Модель swin_base_patch4_window7_224
        """

        transfer_model = timm.create_model('swin_base_patch4_window7_224', pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else "FULL_BLOCK_MODEL"
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model
    
    def _make_swin_base_patch4_window7_224_in22k(self):
        """ Создание модели на базе swin_base_patch4_window7_224_in22k

        Используетя параметр self.pretrained для загрузки предобученной модели.

        Результат
        ---------
        model : torch model
          Модель swin_base_patch4_window7_224
        """

        transfer_model = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else "FULL_BLOCK_MODEL"
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model

    def _make_swin_tiny_patch4_window7_224(self):
        """ Создание модели на базе tiny_patch4_window7_224

        Используетя параметр self.pretrained для загрузки предобученной модели.

        Результат
        ---------
        model : torch model
          Модель tiny_patch4_window7_224
        """

        transfer_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else "FULL_BLOCK_MODEL"
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model
    
    def _make_swin_large_patch4_window12_384(self):
        """ Создание модели на базе tiny_patch4_window7_224

        Используетя параметр self.pretrained для загрузки предобученной модели.

        Результат
        ---------
        model : torch model
          Модель swin_large_patch4_window12_384
        """

        transfer_model = timm.create_model('swin_large_patch4_window12_384', pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else "FULL_BLOCK_MODEL"
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model
    
    def _make_swin_large_patch4_window7_224(self):
        """ Создание модели на базе swin_large_patch4_window7_224

        Используетя параметр self.pretrained для загрузки предобученной модели.

        Результат
        ---------
        model : torch model
          Модель swin_large_patch4_window7_224
        """

        transfer_model = timm.create_model('swin_large_patch4_window7_224', pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else "FULL_BLOCK_MODEL"
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model

    def _make_vit_base_patch16_224(self):
        """ Создание модели на базе swin_large_patch4_window12_384

        Используетя параметр self.pretrained для загрузки предобученной модели.

        Результат
        ---------
        model : torch model
          Модель swin_large_patch4_window12_384
        """

        transfer_model = timm.create_model('vit_base_patch16_224', pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else "FULL_BLOCK_MODEL"
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer,
                                                   first_dropout=False)

        return transfer_model

    def _make_efficientnetv2_s(self):
        """ Создание модели на базе efficientnetv2_s

                Используетя параметр self.pretrained для загрузки предобученной модели.

                Результат
                ---------
                model : torch model
                  Модель efficientnetv2_l
                """

        transfer_model = timm.create_model('tf_efficientnetv2_s', pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else 'blocks.5.13'
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model

    def _make_efficientnetv2_m(self):
        """ Создание модели на базе efficientnetv2_s

                Используетя параметр self.pretrained для загрузки предобученной модели.

                Результат
                ---------
                model : torch model
                  Модель efficientnetv2_l
                """

        transfer_model = timm.create_model('tf_efficientnetv2_m', pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else 'blocks.6.0'
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model

    def _make_efficientnetv2_b1(self):
        """ Создание модели на базе efficientnetv2_b1

                Используетя параметр self.pretrained для загрузки предобученной модели.

                Результат
                ---------
                model : torch model
                  Модель efficientnetv2_b1
                """

        transfer_model = timm.create_model('tf_efficientnetv2_b1', pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else 'blocks.5.5'
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model

    def _make_efficientnetv2_b3(self):
        """ Создание модели на базе efficientnetv2_b3

                Используетя параметр self.pretrained для загрузки предобученной модели.

                Результат
                ---------
                model : torch model
                  Модель efficientnetv2_l
                """

        transfer_model = timm.create_model('tf_efficientnetv2_b3', pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else 'blocks.5.8'
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model

    def _make_resnet18(self):
        """ Создание модели на базе ResNet18

        Используетя параметр self.pretrained для загрузки предобученной модели.

        Результат
        ---------
        model : torch model
          Модель ResNet18
        """

        transfer_model = models.resnet18(pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else 'layer4'
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model

    def _make_vgg16(self):
        """ Создание модели на базе VGG16

            Используетя параметр self.pretrained для загрузки предобученной модели.

            Результат
            ---------
            model : torch model
              Модель VGG16
        """

        transfer_model = models.vgg16(pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else 'features.24'
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model

    def _make_googlenet(self):
        """ Создание модели на базе GoogleNet

            Используетя параметр self.pretrained для загрузки предобученной модели.

            Результат
            ---------
            model : torch model
              Модель GoogleNet
        """

        transfer_model = models.googlenet(pretrained=self.pretrained)
        transfer_model.dropout.p = 0.0
        no_grad_layer = None if self.full_trainable else 'inception5a'
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model

    def _make_alexnet(self):
        """ Создание модели на базе AlexNet

            Используетя параметр self.pretrained для загрузки предобученной модели.

            Результат
            ---------
            model : torch model
              Модель AlexNet
        """

        transfer_model = models.alexnet(pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else 'features.8'
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model

    def make_model(self, name):
        """ Создание модели для трансферного обучения.

        Параметры
        ---------
        name : str
          Название модели. Может принимать значение resnet18, vgg16, alexnet, googlenet или efficientnetv2_b3

        Результат
        ---------
        model : torch model
        """

        name = str(name).lower().strip()

        if name == 'resnet18':
            return self._make_resnet18()
        elif name == 'vgg16':
            return self._make_vgg16()
        elif name == 'alexnet':
            return self._make_alexnet()
        elif name == 'googlenet':
            return self._make_googlenet()
        elif name == 'efficientnetv2_s':
            return self._make_efficientnetv2_s()
        elif name == 'efficientnetv2_m':
            return self._make_efficientnetv2_m()
        elif name == 'efficientnetv2_b1':
            return self._make_efficientnetv2_b1()
        elif name == 'efficientnetv2_b3':
            return self._make_efficientnetv2_b3()
        elif name == 'swin_tiny_patch4_window7_224':
            return self._make_swin_tiny_patch4_window7_224()
        elif name == 'swin_base_patch4_window7_224':
            return self._make_swin_base_patch4_window7_224()
        elif name == 'swin_base_patch4_window7_224_in22k':
            return self._make_swin_base_patch4_window7_224_in22k()
        elif name == 'swin_large_patch4_window12_384':
            return self._make_swin_large_patch4_window12_384()
        elif name == 'swin_large_patch4_window7_224':
            return self._make_swin_large_patch4_window7_224()
        elif name == 'vit_base_patch16_224':
            return self._make_vit_base_patch16_224()
        else:
            raise AttributeError("Параметр name принимает неизвестное значение")

    def __call__(self, name):
        return self.make_model(name=name)
    
    
class PetDataMining:
    """ Генерация признаков для датасета PetFinder с помощью детектора YoloV5
    А так же удаление дубликатов фотографий из датасета.
    
    Метод start создаст и вернет объект self.mining_dataframe_ с дополненными данными
    
    Датасет будет дополнен следующими данными:
    
        n_pets - Кол-во собак и кошек на изображении
        is_unknown - 1, если на изображении не найдены собаки и кошки, иначе 0
        is_cat - 1, если самая точная детекция на фото - кошка, иначе 0
        is_dog - 1, если самая точная детекция на фото - собака, иначе 0
        x_min - минимальное значение по оси x в найденных боксах
        x_max - максимальное значение по оси x в найденных боксах
        y_min - минимальное значение по оси y в найденных боксах
        y_max - максимальное значение по оси y в найденных боксах
        pet_ratio - соотношение размера животных на фото к размеру изображения
    
        У одного фото может существовать только один из флагов is_unknown, is_cat или is_dog
        
    Параметры
    ---------
    petfinder_csv : str
      Путь к датасету в формате csv
    drop_duplicates : bool
      Если True, то будут удалены дубли изображений. По умочалинию False.
    plot_duplicate : bool
      Если True и включено удаление дубликатов, то дуобли изображений будут выводиться
      на экран
    duplicate_thresh : float
      Порог для удаления дублей. По умолчанию 0.9. В этом случае будут удалены изображения,
      которые на 90% схожи по phash.
    plot_detector : bool
      Если True, то будут выведены результаты детекции для каждого изображения
    image_filepath : str
      Путь к папке изображений
    """
    
    def __init__(self, petfinder_csv, drop_duplicates=False, plot_duplicate=False,
                 duplicate_thresh=0.9, plot_detector=False,
                 image_filepath="/kaggle/input/petfinder-pawpularity-score/train/"):
        
        self.petfinder_csv = petfinder_csv
        self.drop_duplicates = drop_duplicates
        self.duplicate_thresh = duplicate_thresh
        self.plot_duplicate = plot_duplicate
        self.plot_detector = plot_detector
        
        self._yolov5x6_model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)
        
        dtype = {
            'Id': 'string',
            'Subject Focus': np.uint8, 'Eyes': np.uint8, 'Face': np.uint8, 'Near': np.uint8,
            'Action': np.uint8, 'Accessory': np.uint8, 'Group': np.uint8, 'Collage': np.uint8,
            'Human': np.uint8, 'Occlusion': np.uint8, 'Info': np.uint8, 'Blur': np.uint8,
            'Pawpularity': np.uint8,
        }
        
        self._petfinder_df = pd.read_csv(petfinder_csv, dtype=dtype)
        
        if image_filepath[-1] != os.path.sep:
            image_filepath = image_filepath + os.path.sep
            
        self._petfinder_df['file_path'] = self._petfinder_df['Id'].apply(lambda x: image_filepath + x + '.jpg')

    @staticmethod
    def _calc_img_hash(file_path):
        """ Вычисления перцептивного хэша изображения
        
        Параметры
        ---------
        file_path : str
          Путь к изображению
          
        Результат
        ---------
        phash : np.array
        """
        
        img = Image.open(file_path)
        img_hash = imagehash.phash(img)
        
        return img_hash.hash.reshape(-1).astype(np.uint8)
    
    def get_images_hash(self):
        """ Вычисляем для каждого изображения phash, создаем колонку phash в self._petfinder_df
        """
        tqdm.pandas()
        self._petfinder_df['phash'] = self._petfinder_df['file_path'].progress_apply(self._calc_img_hash)
    
    def find_similar_images(self, threshold=0.90):
        """ Поиск похожих изображений с помощью измерения
        расстояния Хэмминга между перцептивными хэшами изображений
        
        Расстояние Хэмминга считается как количество отличных битов в хэшах.
        Если все биты будут совпадать, то расстояние Хэмминга, разделенное на длину хэша, будет равно 1.0.
        Такое воспадение можно считать дублем. Эти изображения в целом одинаковые,
        но могут иметь разную цветовую коррекцию.
        Это особенность перцептивного хэша.
        
        Параметры
        ---------
        threshold : float
          Если расстояние Хэмминга двух хешей будет больше порога,
          то данные изображения будут считаться дублями.
          
        Результат
        ---------
        duplicate_idxs : set
          Список дублей изображений
        
        """
        
        assert 'phash' in self._petfinder_df.columns, "Отсутствует хэш для изображений"
        
        # Счеткик для найденных дубликатов, используется для печати
        duplicate_counter = 1
        
        # Индексы дублированных изображений в датафрейме
        duplicate_idxs = set()
        
        # Перебираем каждое изображение в датафрейме
        for idx, phash in enumerate(tqdm(self._petfinder_df['phash'])):
            
            # В данном цикле вычисляем расстояние Хэмминга до каждого иного изображения
            for idx_other, phash_other in enumerate(self._petfinder_df['phash']):
                
                # Вычисляем сходство с помощью расстояния Хэмминга
                similarity = (phash == phash_other).mean()
                
                # Если сравниваются изображения с разными идентификаторами, сходство выше порога и данные изображения
                # нt были ранее обработаны, то добавляем их в список дублей
                if idx != idx_other and similarity > threshold and not(duplicate_idxs.intersection([idx, idx_other])):
                    
                    # Обновляем индекс дублей
                    duplicate_idxs.update([idx, idx_other])
                    
                    # Получаем строки датафрейма с дублями
                    row = self._petfinder_df.loc[idx]
                    row_other = self._petfinder_df.loc[idx_other]
                    
                    # Печать изображений
                    if self.plot_duplicate:
                        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,5))
                        ax[0].imshow(imageio.imread(row['file_path']))
                        ax[0].set_title(f'Idx: {idx}, Pawpularity: {row["Pawpularity"]}')
                        ax[1].imshow(imageio.imread(row_other['file_path']))
                        ax[1].set_title(f'Idx: {idx_other}, Pawpularity: {row_other["Pawpularity"]}')
                        plt.suptitle(f'{duplicate_counter} | PHASH Similarity: {similarity:.3f}')
                        plt.show()
                        
                    # Увеличиваем счетчик дубликатов
                    duplicate_counter += 1
                    
        return duplicate_idxs   
    
    def _drop_duplicate_images(self):
        """ Функция удаляет дубликаты изображений из датафрейма self._petfinder_df
        Функция изменяет self._petfinder_df
        """
        duplicate_idxs = self.find_similar_images(threshold=self.duplicate_thresh)
        self._petfinder_df = self._petfinder_df.drop(duplicate_idxs).reset_index(drop=True)
        
    def get_detector_image_info(self, image_path):
        """ Генерация дополнительной инофрмации для изображения с помощью детектора Yolov5
        
        Какие признаки генерируются:
            n_pets - Кол-во животных на изображении
            labels - Список меток всех найденных животных
            thresholds - список confidence score для каждой метки
            coords - координаты бокса детекции
            x_min - минимальное значение по оси x в найденных боксах
            x_max - максимальное значение по оси x в найденных боксах
            y_min - минимальное значение по оси y в найденных боксах
            y_max - максимальное значение по оси y в найденных боксах
            pet_ratio - соотношение размера животных на фото к размеру изображения
        """
        
        image = imageio.imread(image_path)
        h, w, c = image.shape
        
        if self.plot_detector: # Debug Plots
            fig, ax = plt.subplots(1, 2, figsize=(8,8))
            ax[0].set_title('Детекция', size=16)
            ax[0].imshow(image)        
            
        # Получаем результаты детекции Yolov5 с аугментацией для увеличения качества
        results = self._yolov5x6_model(image, augment=True)
        
        # Маска для пикселей, которые содержат животных. По умолчанию заполняем нулями.
        pet_pixels = np.zeros(shape=[h, w], dtype=np.uint8)
        
        # Словарь для хранения информации об изображении
        h, w, _ = image.shape
        image_info = { 
            'n_pets': 0,
            'labels': [],
            'thresholds': [],
            'coords': [],
            'x_min': 0,
            'x_max': w - 1,
            'y_min': 0,
            'y_max': h - 1,
        }
        
        # Список найденных боксов и меток животных для печати
        pets_found = []
        
        # Сохраняем инофрмацию по каждой детекции
        for x1, y1, x2, y2, treshold, label in results.xyxy[0].cpu().detach().numpy():
            
            label = results.names[int(label)]
            
            # Предполагаем, что на всех фотографиях у нас либо коты, либо собаки
            if label in ['cat', 'dog']:
                
                image_info['n_pets'] += 1
                image_info['labels'].append(label)
                image_info['thresholds'].append(treshold)
                image_info['coords'].append(tuple([x1, y1, x2, y2]))
                image_info['x_min'] = max(x1, image_info['x_min'])
                image_info['x_max'] = min(x2, image_info['x_max'])
                image_info['y_min'] = max(y1, image_info['y_min'])
                image_info['y_max'] = min(y2, image_info['y_max'])
                
                # Устанавливаем пикель в 1 в местах детекции животных
                pet_pixels[int(y1):int(y2), int(x1):int(x2)] = 1
                
                # Добавляем в список боксов найденное животное
                pets_found.append([x1, x2, y1, y2, label])
    
        if self.plot_detector:
            for x1, x2, y1, y2, label in pets_found:
                c = 'red' if label == 'dog' else 'blue'
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=c, facecolor='none')
                ax[0].add_patch(rect)
                ax[0].text(max(25, (x2+x1)/2), max(25, y1-h*0.02), label, c=c, ha='center', size=14)
                    
        # Вычисляем соотношение размера животных на фото к размеру изображения
        image_info['pet_ratio'] = pet_pixels.sum() / (h*w)
    
        if self.plot_detector:
            ax[1].set_title('Pet Ratio', size=16)
            ax[1].imshow(pet_pixels)
            plt.show()
            
        return image_info        
        
    def detector_data_mining(self):
        """ Майнинг данных для всех изображений с помощью детектора Yolov5
        
        Датафрейм self._petfinder_df не изменяется, возвращается модифицированная копия
        Добавляются столбцы n_pets, x_min, x_max, y_min, y_max, pet_ratio
        
        Используется метод self._get_detector_image_info() 
        
        Результат
        ---------
        mining_dataframe : pd.DataFrame
          Дополненный датафрейм self._petfinder_df
        """
        
        # Image Info
        images_info = {
            'n_pets': [],
            'pet_label': [],
            'x_min': [],
            'x_max': [],
            'y_min': [],
            'y_max': [],
            'pet_ratio': [],
        }
        
        for idx, file_path in enumerate(tqdm(self._petfinder_df['file_path'])):
            
            image_info = self.get_detector_image_info(file_path)
            
            images_info['n_pets'].append(image_info['n_pets'])
            images_info['x_min'].append(image_info['x_min'])
            images_info['x_max'].append(image_info['x_max'])
            images_info['y_min'].append(image_info['y_min'])
            images_info['y_max'].append(image_info['y_max'])
            images_info['pet_ratio'].append(image_info['pet_ratio'])
            
            # Not Every Image can be Correctly Classified
            labels = image_info['labels']
            if len(set(labels)) == 1: # unanimous label
                images_info['pet_label'].append(labels[0])
            elif len(set(labels)) > 1: # Get label with highest confidence
                images_info['pet_label'].append(labels[0])
            else: # unknown label, yolo could not find pet
                images_info['pet_label'].append('unknown') 
                      
        mining_dataframe = self._petfinder_df.copy()
                
        for k, v in images_info.items():
            mining_dataframe[k] = v  
            
        mining_dataframe = self._post_processing(mining_dataframe)
            
        return mining_dataframe
    
    @staticmethod
    def _post_processing(mining_dataframe):
        """ Потсобрабтка датафрейма
        """
        mining_dataframe["is_unknown"] = mining_dataframe["pet_label"].apply(lambda x: 1 if x == 'unknown' else 0)
        mining_dataframe["is_dog"] = mining_dataframe["pet_label"].apply(lambda x: 1 if x == 'dog' else 0)
        mining_dataframe["is_cat"] = mining_dataframe["pet_label"].apply(lambda x: 1 if x == 'cat' else 0)

        return mining_dataframe
        
            
    def start(self):
        """ Запуск майнинга данных
        
        Результат
        ---------
        self.mining_dataframe_ : pd.DataFrame
        """
        
        if self.drop_duplicates:
            self.get_images_hash()
            self._drop_duplicate_images()
        
        self.mining_dataframe_ = self.detector_data_mining()
        
        return self.mining_dataframe_
    
    def __call__(self):
        return self.start()


class PetDataSet(Dataset):
    """ Датасет с набором данных PetFinder методом KFOLD

    Параметры
    ---------
    pet_df : pandas dataframe
      Общий датафрейм с данными соревнования (обучение + тест).
      Датафрейм должен обязательно содержать следующие колонки:
      - Pawpularity (целевая переменная)
      - Id (хэш файла изображения)
      - label (класс изображения. Может быть расчитан как целое число Pawpularity или по другому алгоритму,
               класс может использоваться для получения веса экземляра и метки в формате вектора нормального распределения)
      - split (используется для разделения данных по выборкам,
               принимает значения train, val или predict)
    train_photo_dir : str
      Путь к папке с фото для обучения.
    test_photo_dir : str
      Путь к папке с фото для теста.
    image_size : int
      Размер выходного изображения для генерации датасета.
      По умолчанию 224.
    n_splits : int
      Количество фолдов для разбиения датасета. По умолчанию 10.
    fold_number : int
      Номер фолда для генератора объектов. По умолчанию 0.
    class_counts : int
      Кол-во классов, по умолчанию None. Используется для генерации метки в формате вектора нормального распределения.
      Если None, то метка не будет сформирована.
    train_augmentation : bool
      Если True, то к обучающему датасету будет применяться аугментация. По умолчанию True.
    val_augmentation : bool
      Если True, то к тестовой и валидацитонной выборкам будет применена аугментация
    class_weights : list
      Список весов классов. Количество классов определяется вне данного класса. Данн
    pca_for_add_features : bool
      Требуется ли преобразовать дополнительные признаки с помощью PCA. Количество главных компонент
      равно кол-ву признаков. Сжатия пространства нет.
    gaussian_sigma : float
      СКО для формулы распределения Гаусса. Используется для создания меток в формате вектора
      в виде нормального распределения со средним в значении метки класса. Требуется для обучения модели
      с функцией потерь KLDiv. По умолчанию 2.
    seed : int
      Фиксация генератора случайных чисел.
    """

    def __init__(self, pet_df, class_counts=100, image_size=224, train_augmentation=True, val_augmentation=False, class_weights=None,
                 pca_for_add_features=True, std_for_add_features=True, gaussian_sigma=2.0, p_vflip=0.5, p_hflip=0.5, seed=None):

        if seed is not None:
            set_seed(seed)

        self._seed = seed
        
        self.__detector_add_features = ['n_pets', 'is_unknown', 'is_cat', 'is_dog', 'x_min', 'x_max', 'y_min', 'y_max', 'pet_ratio']
        self.__standard_add_features = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
                                         'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']

        assert 'Pawpularity' in pet_df.columns, "Датафрейм pet_df должен содержать колонку Pawpularity"
        assert 'Id' in pet_df.columns, "Датафрейм pet_df должен содержать колонку Id"
        assert 'label' in pet_df.columns, "Датафрейм pet_df должен содержать колонку Id"
        assert 'photo_dir' in pet_df.columns, "Датафрейм pet_df должен содержать колонку photo_dir"
        
        if set(self.__detector_add_features).issubset(set(pet_df.columns)):
            self._additive_features_names = self.__standard_add_features + self.__detector_add_features
        else:
            assert all([f_name not in pet_df.columns for f_name in self.__detector_add_features]), "В датасете присутсвует неполное кол-во колонок признаков детекции"
            self._additive_features_names = self.__standard_add_features
            
        self._gaussian_sigma = gaussian_sigma
        self._class_counts = class_counts
        self._class_weights = class_weights
        self._pca_for_add_features = pca_for_add_features
        self._std_for_add_features = std_for_add_features

        # Средние значения для IMAGE NET, используются при стандартизации изображений
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        # СКО для IMAGE NET, используются при стандартизации изображений
        IMAGENET_STD = [0.229, 0.224, 0.225]

        self._pet_df = pet_df

        # Определяем функцию обработки изображений для валидации и теста
        if train_augmentation:
            self._train_transforms = T.Compose(
                [T.Resize([image_size, image_size]),
                 T.RandomHorizontalFlip(p=p_hflip),
                 T.RandomVerticalFlip(p=p_vflip),
                 #T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                 T.RandomErasing(p=0.5, scale=(0.02,0.1), ratio=(0.3,3.3)),
                 #T.RandomAffine(degrees=10, translate=(0, 0.2), scale=(0.9, 1), shear=(6, 9)),              
                 T.ConvertImageDtype(torch.float32),
                 T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                 ]
            )
        else:
            self._train_transforms = T.Compose(
                [T.Compose([T.Resize([image_size, image_size])]),
                 T.ConvertImageDtype(torch.float32),
                 T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                 ]
            )      
            
        if val_augmentation:
            self._val_transforms = T.Compose(
                [T.Resize([image_size, image_size]),
                 T.RandomHorizontalFlip(p=p_hflip),
                 T.RandomVerticalFlip(p=p_vflip),         
                 T.ConvertImageDtype(torch.float32),
                 T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                 ]
            )
        else:
            self._val_transforms = T.Compose(
                [T.Compose([T.Resize([image_size, image_size])]),
                 T.ConvertImageDtype(torch.float32),
                 T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                 ]
            )      

        self._train_df = pet_df[pet_df.split == 'train'].copy()
        self._train_size = len(self._train_df)
        self._val_df = pet_df[pet_df.split == 'val'].copy()
        self._val_size = len(self._val_df)
        self._test_df = pet_df[pet_df.split == 'predict'].copy()
        self._test_size = len(self._test_df)
        
            
        if self._std_for_add_features:
            self._make_std_features()        

        if self._pca_for_add_features:
            self._make_pca_features()

        self._lookup_dict = {
            'train': (self._train_df, self._train_size, self._train_transforms),
            'val': (self._val_df, self._val_size, self._val_transforms),
            'predict': (self._test_df, self._test_size, self._val_transforms),
        }

        # По умолчанию включаем режим обучения
        self.set_split('train')

    def _make_pca_features(self):
        """ Преобразование дополнительных признаков с помощью PCA

        Параметры
        skf_dict : словарь
          Словарь с фолдами обучающего датасета и тренировочного
          Формат: {номер фолда: (датасет обучения, датасет валидации)}

        Результат
        ---------
        std_skf_dict : dict
          Словарь с фолдами со стандартизованными доп. признаками
        """

        features_train_df = self._train_df.loc[:, self._additive_features_names]
        features_val_df = self._val_df.loc[:, self._additive_features_names]
        features_test_df = self._test_df.loc[:, self._additive_features_names]

        pca = PCA(random_state=self._seed)
        pca_features_train_df = pca.fit_transform(features_train_df)
        pca_features_val_df = pca.transform(features_val_df)
        pca_features_test_df = pca.transform(features_test_df)

        for idx, feature_name in enumerate(self._additive_features_names):
            self._train_df.loc[:, feature_name] = pca_features_train_df[:, idx]
            self._val_df.loc[:, feature_name] = pca_features_val_df[:, idx]
            self._test_df.loc[:, feature_name] = pca_features_test_df[:, idx]
            
    def _make_std_features(self):
        """ Преобразование дополнительных признаков с помощью стандартизации

        Параметры
        skf_dict : словарь
          Словарь с фолдами обучающего датасета и тренировочного
          Формат: {номер фолда: (датасет обучения, датасет валидации)}

        Результат
        ---------
        std_skf_dict : dict
          Словарь с фолдами со стандартизованными доп. признаками
        """

        features_train_df = self._train_df.loc[:, self._additive_features_names]
        features_val_df = self._val_df.loc[:, self._additive_features_names]
        features_test_df = self._test_df.loc[:, self._additive_features_names]

        scaler = StandardScaler()
        std_features_train = scaler.fit_transform(features_train_df)
        std_features_val = scaler.transform(features_val_df)
        std_features_test = scaler.transform(features_test_df)

        for idx, feature_name in enumerate(self._additive_features_names):
            self._train_df.loc[:, feature_name] = std_features_train[:, idx]
            self._val_df.loc[:, feature_name] = std_features_val[:, idx]
            self._test_df.loc[:, feature_name] = std_features_test[:, idx]

    def set_split(self, split='train'):

        split = str(split).strip().lower()
        assert split in ('train', 'val', 'predict'), "split может принимать значения train или val"

        self._target_split = split
        self._target_df, self._target_size, self._target_transforms = self._lookup_dict[split]

    @staticmethod
    def _gaussian_label(label, sigma=2.0, class_size=100):
        """ Создание метки в формате вектора с нормальным распределением значений

        Вектор содержит в себе вероятность принадлежности объекта к определенному классу.
        Максимальная вероятность (пик колокола нормального распределения) будет в значении класса.

        Параметры
        ---------
        label : int
          Номер класса
        sigma : float
          СКО для формулы Гаусса
        class_size : int
          Общее количество классов

        Результат
        ---------
        label : numpy array
          Метка в формате вектора с нормальным распределением значений
        """

        assert 0 <= label < class_size, f"Значение метки label ({label}) не входит в диапазон от 0 до {class_size}"

        k = np.arange(0, class_size)
        label = np.exp(-(k - label) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)

        # Проверку убрал, так как если label лежит у границы, то там не получается сумма 1
        # assert np.sum(label) == 1.0, "Сумма вероятностей метки функции Гаусса не равна 1"

        return label

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """ Возвращает элемент датасета в формате:

        {
        'hash': хэш изображения,
        'x_data': Тензор изображения размером N x C x H x W,
        'additional_features': дополнительные признаки,
        'image_path': путь к файлу изображения,
        'pawpularity': оценка привлекательности изображения,
        'class_weight': вес класса изображения (есть только при наличии данных в self._class_weights),
        'label': метка в формате вектора в виде нормального распределения со средним в значении метки класса
                 (формируется, если задано self._class_counts и строка имеет поле label)
        }

        Если датасет работает в режиме predict, то возвращаются только hash, x_data и image_path
        """

        # Получаем строку датафрейма по его индексу
        row = self._target_df.iloc[index, :]

        # Словарь, который будем возвращать, инициализируем его значением хэша изображения
        model_data = {'hash': row.Id}

        # Формируем ссылку к изображению и проверяем на доступность файла
        if row.photo_dir[-1] != os.sep:
            row.photo_dir[-1] = os.sep
        image_path = str(row.photo_dir) + str(row.Id) + '.jpg'
        if not os.path.exists(image_path):
            raise Exception(f"Файл {image_path} не существует")

        # Читаем изображение и применяем функцию обработки изображения
        img_source = read_image(image_path)
        x_data = self._target_transforms(img_source)

        # Добавляем в возвращаемый словарь обработанное изображение
        model_data['x_data'] = x_data
        # Добавляем в возвращаемый словарь путь к исходному изображению
        model_data['image_path'] = image_path
        
        # Добавляем дополнительные признаки
        model_data['additional_features'] = torch.tensor(row[self._additive_features_names].values.astype(float),
                                                             dtype=torch.float32)

        # Если работает режим прежсказания, то возвращаем model_data только с ключами x_data, hash и additional_features
        if self._target_split == 'predict':
            return model_data

        # Получаем значение популярности
        model_data['pawpularity'] = torch.tensor(row.Pawpularity).type(dtype=torch.float32)

        # Если в классе инициализированы веса, то формируем class_weight
        if self._class_weights is not None:
            model_data['class_weight'] = self._class_weights[row.label]

        # Если задано количество классов и строка имеет метку класса в поле label, то формируем
        # label в виде вектора нормального распределения
        if self._class_counts is not None and getattr(row, 'label', None) is not None:
            model_data['label_distribution'] = self._gaussian_label(label=row.label, sigma=self._gaussian_sigma,
                                                       class_size=self._class_counts)
            model_data['label_source'] = row.label

        return model_data


class PetDataLoader(pl.LightningDataModule):

    def __init__(self, pet_dataframe, train_loader_params=None, val_loader_params=None, test_loader_params=None,
                 dataset_params=None,
                 seed=None):

        super().__init__()

        if seed is not None:
            set_seed(seed)

        self._pet_dataframe = pet_dataframe

        if not train_loader_params:
            train_loader_params = {
                'batch_size': 64,
                'shuffle': True,
                'num_workers': 2,
                'drop_last': True,
            }

        if not val_loader_params:
            val_loader_params = {
                'batch_size': 64,
                'shuffle': False,
                'num_workers': 2,
                'drop_last': False
            }

        if not test_loader_params:
            test_loader_params = {
                'batch_size': 64,
                'shuffle': False,
                'num_workers': 2,
                'drop_last': False
            }

        if not dataset_params:
            dataset_params = {
                'class_counts': 100,
                'image_size': 224,
                'train_augmentation': True,
                'val_augmentation': False,
                'p_vflip': 0.5,
                'p_hflip': 0.5,
                'class_weights': None,
                'pca_for_add_features': True,
                'gaussian_sigma': 2.0,
                'seed': None,
            }

        self.train_loader_params = train_loader_params
        self.val_loader_params = val_loader_params
        self.test_loader_params = test_loader_params
        self.dataset_params = dataset_params

        self.make_split_dict()

    def make_split_dict(self):

        self.train_dataset = PetDataSet(pet_df=self._pet_dataframe, **self.dataset_params)
        self.train_dataset.set_split('train')

        self.val_dataset = PetDataSet(pet_df=self._pet_dataframe, **self.dataset_params)
        self.val_dataset.set_split('val')

        self.predict_dataset = PetDataSet(pet_df=self._pet_dataframe, **self.dataset_params)
        self.predict_dataset.set_split('predict')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_loader_params['batch_size'],
                          shuffle=self.train_loader_params['shuffle'], drop_last=self.train_loader_params['drop_last'],
                          num_workers=self.train_loader_params['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_loader_params['batch_size'],
                          drop_last=self.val_loader_params['drop_last'], shuffle=self.val_loader_params['shuffle'],
                          num_workers=self.val_loader_params['num_workers'])

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.test_loader_params['batch_size'],
                          drop_last=self.test_loader_params['drop_last'], shuffle=self.test_loader_params['shuffle'],
                          num_workers=self.test_loader_params['num_workers'])
    
    def static_tta_dataloaders(self, mode="val"):
        """ Создание загрузчиков для TTA
        
        Параметры
        ---------
        mode : str
          Режим данных. Принимает значение train, val или predict
        
        Результат
        ---------
        loader_1, loader_2, loader_3, loader_4 : DataLoader
          loader_1 не делает поворотов
          loader_2 делает горизонатльный разворот
          loader_3 делает вертикальны йразворот
          loader_4 делает поворот по вертикали и горизонтали
        """
        
        mode = str(mode).lower().strip()
        assert mode in ('train', 'val', 'predict')
        
        dataset_params_1 = copy(self.dataset_params)
        dataset_params_1['val_augmentation'] = True
        dataset_params_1['p_vflip'] = 0.0
        dataset_params_1['p_hflip'] = 0.0
        dataset_params_2 = copy(self.dataset_params)
        dataset_params_2['val_augmentation'] = True
        dataset_params_2['p_vflip'] = 1.0
        dataset_params_2['p_hflip'] = 0.0     
        dataset_params_3 = copy(self.dataset_params)
        dataset_params_3['val_augmentation'] = True
        dataset_params_3['p_vflip'] = 0.0
        dataset_params_3['p_hflip'] = 1.0  
        dataset_params_4 = copy(self.dataset_params)
        dataset_params_4['val_augmentation'] = True
        dataset_params_4['p_vflip'] = 1.0
        dataset_params_4['p_hflip'] = 1.0
        
        dataset_1 = PetDataSet(pet_df=self._pet_dataframe, **dataset_params_1)
        dataset_2 = PetDataSet(pet_df=self._pet_dataframe, **dataset_params_2)
        dataset_3 = PetDataSet(pet_df=self._pet_dataframe, **dataset_params_3)
        dataset_4 = PetDataSet(pet_df=self._pet_dataframe, **dataset_params_4)
        
        dataset_1.set_split(mode)
        dataset_2.set_split(mode)
        dataset_3.set_split(mode)
        dataset_4.set_split(mode)          
            
        loader_1 = DataLoader(dataset_1, batch_size=1, drop_last=False, shuffle=False,
                              num_workers=2)
        loader_2 = DataLoader(dataset_2, batch_size=1, drop_last=False, shuffle=False,
                              num_workers=2)
        loader_3 = DataLoader(dataset_3, batch_size=1, drop_last=False, shuffle=False,
                              num_workers=2)
        loader_4 = DataLoader(dataset_4, batch_size=1, drop_last=False, shuffle=False,
                              num_workers=2)    
        
        return loader_1, loader_2, loader_3, loader_4
            

    @staticmethod
    def _make_label_for_pet_df(pet_df, use_kldiv=False):
        """ Создание новой колонки Label.
        В данной колонке находится новый класс изображения, необходимый для расчета весов.
        
        Если use_kldiv = True, то классов будет 100 штук. Каждая оценка - один класс.
        Если False, то классов будет 20. Например, оценка от 1 до 5 - класс 1, от 5 до 10 - класс 2.
        
        Результат
        ---------
        pet_df, class_count : pd.DataFrame, int
        """

        pet_df = pet_df.copy()

        # Выделяем номер класса изображения, если включен параметр use_kldiv
        if use_kldiv:
            class_counts = 100
            # Номер класса равен целому числу метрики популярности
            pet_df['label'] = pet_df['Pawpularity'].astype(int)
            # Набор уникальных номеров классов
            uniq_labels = pet_df['label'].dropna().drop_duplicates().values

            # Предполагаем, что в uniq_labels нет нулевого класса
            # Далее, мы будем сдвигать нумерацию класса на единицу влево.
            # Если 0 уже есть, то это ошибка.
            if 0 in uniq_labels:
                raise Exception(f"В обучающем датасете присутствует метка с кодом 0")

            # Проверяем есть ли все номера классов в датасете от 1 до 100 включительно
            # Если хоть один класс не будет найден, то это ошибка.
            for label in np.arange(1, 101):
                if label not in uniq_labels:
                    raise Exception(f"В обучающем датасете отсутствует метка с кодом {label}")

            # Если все проверки завершены успешно, то сдвигаем метки классов на единицу влево
            # Теперь классы имеют значения от 0 до 99 включительно
            pet_df['label'] = pet_df['label'] - 1

        else:
            # Выделяем номер класса изображения, если выключен параметр use_kldiv
            # Разбиваем на 10 классов. Класс берется в зависимости от значения популярности от 0 до 100 с шагом в 10.
            # Такая классификация имеет смысл для использования весов в функции ошибок RMSE.
            class_counts = 20
            pet_df['label'] = pd.cut(pet_df.Pawpularity, bins=class_counts, labels=np.arange(class_counts))

        return pet_df, class_counts

    @staticmethod
    def _get_weights(pet_df, class_counts):
        """ Рассчет весов для каждого изображения в датафрейме
        на основе колонки label.
        
        Результат
        ---------
        class_weights : dict
          Словарь в формате {класс: вес}
        """

        # Считаем кол-во экземпляров в каждом классе для расчета весов классов
        class_stats = pet_df[pet_df.split == 'train'].groupby('label').count()['Pawpularity']
        # Находим веса для каждого класса, веса не нормализованы. Класс с самым больших кол-вом экземпляров имеет вес 1,
        # остальные классы имеют веса больше 1 пропорционально их размеру относительно самого крупного класса
        class_weights = class_stats.max() / class_stats
        # Создаем словарь для классов в формате {номер класса: вес класса}
        class_weights = dict(zip(list(class_weights.index), list(class_weights.values)))

        assert len(class_weights) == class_counts, "Размер словаря с весами не равен количеству классов в датасете"

        return class_weights

    @classmethod
    def create_kfold_loaders(cls, train_csv, test_csv, train_photo_dir, test_photo_dir, n_splits=10, size_dataset=None,
                              use_kldiv=False, train_loader_params=None, val_loader_params=None, test_loader_params=None,
                              dataset_params=None, seed=None):
        
        """ Инициализация нескольких экземпляров класса из файлов соревнования на Kaggle для каждого фолда.
        
        Обучающий датасет будет разбит на фолды исходя из их количества в параметре n_splits.
        Каждый фолд будет дополнительно разделен на обучающую и валидационную части. 
        Данные из тестового набора будут определены для набора предсказания и одинаковы для каждого фолда.

        Функция добавляет дополнительное поле label, которое сожержит номер класса.
        Классификация определяется в зависимости от параметра use_kldiv.

        Параметры
        ---------
        train_csv : str
          Пусть к файлу train.csv
        test_csv : str
          Пусть к файлу val.csv
        train_photo_dir : str
          Путь к папке с фото для обучения
        test_photo_dir : str
          Путь к папке с фото для теста
        n_splits : int
          Количество фолдов
        use_kldiv : bool
          Настройка классификации для использования функции потерь KLDiv.
          Если True, то общее кол-во классов равно 100. И номер класса равен целому числу метрики популярности.
          Дальше эта метка непосредственно в классе будет преобразована в вектор с нормальным распределением данных.
          Если False, то все данные бьются на 10 классов со значением популярности от 0 до 100 с шагом 10.
        train_loader_params : dict
          Словарь параметров загрузчика для обучения. См. код инициализации класса.
        val_loader_params : dict
          Словарь параметров загрузчика для валидации. См. код инициализации класса.
        test_loader_params : dict
          Словарь параметров загрузчика для теста/предсказания. См. код инициализации класса.
        dataset_params : dict
          Словарь параметров инициализации класса PetDataset. 
          См. код инициализации класса PetDataLoader и документацию PetDataset.
        size_dataset : int
          Ограничитель размера датасета. Если None, то будут использованы данные на основе всего датасета.
          Если задано число, то размер датасета будет ограничен этим количеством.
        seed : int
          Фиксация генератора случайных чисел.


        Результат
        ---------
        dataset : PetDataset object
        """        

        if seed is not None:
            set_seed(seed)

        pet_train_df = pd.read_csv(train_csv)
        pet_train_df, class_counts = cls._make_label_for_pet_df(pet_df=pet_train_df, use_kldiv=use_kldiv)

        # Определяем финальную длину данных
        if size_dataset:
            n_total = size_dataset
        else:
            n_total = len(pet_train_df)

        pet_train_df = pet_train_df.iloc[np.random.permutation(len(pet_train_df))]
        pet_train_df = pet_train_df.iloc[:n_total]
        pet_train_df.loc[:, 'photo_dir'] = train_photo_dir
        pet_train_df.loc[:, 'split'] = None

        # Считываем датасет для теста
        pet_test_df = pd.read_csv(test_csv)
        # Для всего тестового набора устанавливаем split = predict
        pet_test_df['split'] = 'predict'
        pet_test_df['photo_dir'] = test_photo_dir
        pet_test_df['Pawpularity'] = None

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        skf.get_n_splits(pet_train_df, pet_train_df['label'])

        kfold_dataloaders = []

        for fold_index, (train_index, val_index) in enumerate(skf.split(pet_train_df, pet_train_df['label'])):

            pet_df_fold_copy = pet_train_df.copy()
            split_col_index = np.where(pet_df_fold_copy.columns == 'split')[0]
            pet_df_fold_copy.iloc[train_index, split_col_index] = 'train'
            pet_df_fold_copy.iloc[val_index, split_col_index] = 'val'

            if dataset_params is not None:
                class_weights = cls._get_weights(pet_df=pet_df_fold_copy, class_counts=class_counts)
                dataset_params['class_weights'] = class_weights

            # Объединяем все данные в единый датафрейм
            pet_df_fold_copy = pd.concat([pet_df_fold_copy, pet_test_df], axis=0)

            kfold_dataloader = cls(pet_dataframe=pet_df_fold_copy, train_loader_params=train_loader_params,
                                   val_loader_params=val_loader_params, test_loader_params=test_loader_params,
                                   dataset_params=dataset_params, seed=seed)

            kfold_dataloaders.append(kfold_dataloader)

        return kfold_dataloaders


class PetFinderTransferModel(pl.LightningModule):
    """ Модель для трансферного обучения с помощью ошибки RMSE

    Параметры
    ---------
    model_name : str
      Название модели. Допустимые значения см. в документации TransferNet
      По умолчанию resnet18.
    output_dims : list
      Структура полносвязной сети на выходе экстратора признаков.
      Длина списка - количество полносвязных слоев.
      Значения списка - кол-во нейронов в каждом полносвязном слое.
      Пример: [1024, 512, 256, 1] - четыре полносвязных слоя.
              Первый слой с 1024 нейронами, второй слой с 512 нейронами,
              третий слой с 256 нейронами и последний слой с 1 нейроном.
      По умолчанию [128, 256, 1].
      !!! Последний слой должен содержать только 1 нейрон.
    full_trainable : bool
      Сделать ли модель полностью обучаемой. По умолчанию False. Если False, то для обучения доступна только
      полносвязная сеть и около 10% окончания экстрактора признаков
    dropout : float
      Значение вероятности для всех слоев Dropout полносвязной сети.
      По умолчанию 0.5
    learning_rate : float
      Скорость обучения. По умолчанию 0.01
    l2_regularization : float
      Регуляризация весов L2. По умолчанию 0.001
    adam_betas : tuple
      Коэффициенты оптимизатора Adam. По умолчанию (0.9, 0.999)
    pretrained : bool
      Если True, то будут загружена предобученная модель на ImageNet.
      По умолчанию True.
      !!! ВАЖНО: Класс не тестировался на не предобученных моделях.
          Структура модели при значении pretrained=False может отличаться, поведение класса может быть некорректным.
    use_weights : bool
      Флаг использования весов классов в функции ошибок. По умолчанию False.
    plot_epoch_loss : bool
      Флаг печати графиков ошибок в конце каждой эпохи во время обучения. По умолчанию True.
    seed : int
      Фиксация генератора случайных чисел.
    """

    def __init__(self, model_name='resnet18', output_dims=None, dropout=0.5, learning_rate=0.01, full_trainable=False,
                 l2_regularization=1e-3, adam_betas=(0.9, 0.999), pretrained=True, use_weights=False,
                 plot_epoch_loss=True, seed=None):
        super().__init__()

        if seed is not None:
            set_seed(seed)

        if not output_dims:
            output_dims = [128, 256, 1]

        self._check_output_dims(output_dims)

        # Здесь будет находиться модель для извлечения признаков
        self.feature_extractor = None
        # Здесь будет находиться модель полносвязной сети, вход - данные из экстрактора признаков
        self.head_fc = None
        # Определяем feature_extractor и head_fc
        self._make_model_layers(model_name=model_name, output_dims=output_dims, dropout=dropout,
                                full_trainable=full_trainable, pretrained=pretrained)

        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.adam_betas = adam_betas
        self.use_weights = use_weights

        # Словарь для хранения значения ошибок на стадии обучения и валидации
        # Для значений типа train добавляем значение np.nan, так как при первом запуске модель вначале осуществляет
        # шаг валидации без обучения и добавляет значения в списки типа val. Это будет считаться эпохой №0.
        self.train_history = {
            'train_loss': [np.nan],
            'train_rmse': [np.nan],
            'val_loss': [],
            'val_rmse': [],
        }
        self.plot_epoch_loss = plot_epoch_loss

        self.save_hyperparameters()

    @staticmethod
    def _check_output_dims(output_dims):
        """Проверка корректности размера последнего полносвязного слоя"""
        assert output_dims[-1] == 1, f"Кол-во нейронов в последнем слое output_dims должен быть равен 1. Текущее значение: {output_dims}"

    @staticmethod
    def _get_model_fc_layer_name(model):
        """ Определяем название полносвязного слоя у модели

        Параметры
        ---------
        model : TransferNet object

        Результат
        ---------
        fc_name : str
          Название полносвязного слоя модели
        """

        # В зависимости от модели полносвязный слой может называться по-разному
        # Находим имя полносвязной сети
        if 'fc' in model.__dir__():
            fc_name = 'fc'
        elif 'head' in model.__dir__():
            fc_name = 'head'
        elif 'classifier' in model.__dir__():
            fc_name = 'classifier'
        else:
            raise Exception("В модели не найден полносвязный слой")

        return fc_name

    def _make_model_layers(self, model_name, output_dims, dropout, full_trainable, pretrained):
        """ Загрузка модели и настройка переменных self.feature_extractor и self.head_fc

        Параметры
        ----------
        model_name : str
        output_dims : list
        dropout : float
        pretrained : bool

        Более подробное описание параметров в описании класса.
        """

        model = TransferNet(output_dims=output_dims, dropout=dropout, full_trainable=full_trainable,
                            pretrained=pretrained)(name=model_name)

        """ В этом блоке выделим в отдельные переменные экстрактор признаков модели и полносвязную сеть.
        Так как может возникнуть задача добавления к признакам сверточной сети дополнительных признаков из вне.
        Так как меняется размер признаков, необходимо изменить размер входа полносвязной сети.
        В модели self.model размер входа полносвязной сети равен выходу стандартного экстрактора признаков
        """
        # В зависимости от модели полносвязный слой может называться по-разному
        # Находим имя полносвязной сети
        fc_name = self._get_model_fc_layer_name(model)

        # Копируем в отдельную переменную полносвязную сеть
        self.head_fc = getattr(model, fc_name)

        # А в исходной модели в полносвязный слой устанавливаем пустую последовательность.
        # Таким образом теперь self.model работает как экстрактор признаков из изображения
        setattr(model, fc_name, nn.Sequential())

        self.feature_extractor = model

    def _rmse_loss(self, predict, target, weights=None):
        """ Функция ошибок RMSE, может учитывать веса

        Параметры
        ---------
        predict : array
          Список предсказанных значений
        target : array
          Список целевых значений
        weights : array
          Список весов для кажого элемента. По умолчанию None.
          Если задан список весов, то ошибка каждого предсказанного значения будет умножена на вес.

        Результат
        ---------
        rmse : float
          Ошибка rmse
        """

        # Параметр reduction возвращает результат MSE каждому элементу батча
        loss = F.mse_loss(predict, target, reduction='none')

        # Если указаны веса, умножаем на них ошибки
        if weights is not None and self.use_weights:
            loss = loss * weights

        # Возвращаем среднее значение RMSE по батчу
        return torch.sqrt(torch.mean(loss))

    def forward(self, x_in):
        features = self.feature_extractor(x_in['x_data'])
        x_out = self.head_fc(features).squeeze()
        return x_out

    def configure_optimizers(self):
        """Конфигурация оптимизатора и планировщика скорости обучения"""
        optimizer = optim.AdamW(self.parameters(), betas=self.adam_betas, lr=self.learning_rate,
                                weight_decay=self.l2_regularization)
        sheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                  T_0=20,
                                                                  eta_min=1e-4)

        return [optimizer], [sheduler]

    def _share_step(self, batch, batch_idx, mode='train'):
        """ Общий шаг для обучения, валидации и теста

        Параметры
        ---------
        batch : dict
          Батч-словарь в следующем формате:
          {
          'hash': список хэшей изображения,
          'x_data': список тензоров изображения размером N x C x H x W,
          'image_path': список путей к файлам изображений,
          'pawpularity': список метрик популярности изображения (может отсутствовать, см. описание класса PetDataset),
          'class_weight': список списков  весов класса изображения (может отсутствовать, см. описание класса PetDataset),
          'label': список меткок в формате вектора в виде нормального распределения со средним в значении метки класса
                   (может отсутствовать, см. описание класса PetDataset)
          }
        batch_idx : int
          Номер батча
        mode : str
          Режим. Используется только для префикса названий ошибок в логе.
          По умолчанию train
        """

        # Получаем предсказанную популярность для батча
        pred_pawpularity = self(batch)

        target_pawpularity = batch['pawpularity']
        weights = batch.get('class_weight', None)

        # Считаем ошибку RMSE без учета весов и логируем ее
        rmse_loss = self._rmse_loss(predict=pred_pawpularity, target=target_pawpularity)
        self.log(f'rmse_{mode}_loss', rmse_loss, prog_bar=True)

        # Если задан режим весов, то вычисляем взвешенную RMSE. В этом случае оптимизируемый loss будет равен
        # взвешенной ошибке, иначе будет равен обычному RMSE.
        if self.use_weights:
            weighted_rmse_loss = self._rmse_loss(predict=pred_pawpularity, target=target_pawpularity, weights=weights)
            loss = weighted_rmse_loss
        else:
            loss = rmse_loss

        # Возвращаем loss и rmse_loss без учета весов. Если режим весов выключен, то loss = rmse_loss
        # Таким образом мы можем на графике смотреть и общую ошибку, которая может быть взвешенной,
        # так и обычную RMSE без весов, которая учитывается в соревновании
        return {'loss': loss, 'rmse_loss': rmse_loss.detach()}

    def training_step(self, batch, batch_idx):
        """Шаг обучения"""
        return self._share_step(batch, batch_idx, mode='train')

    def training_epoch_end(self, outputs):
        """Действия после окончания каждой эпохи обучения

        Параметры
        ---------
        outputs : list
          Список словарей. Каждый словарь - результат функции self._share_step для определенного батча на шаге обучения
        """

        # Считаем средние ошибки loss и rmse_loss по эпохе
        avg_train_loss = torch.tensor([x['loss'] for x in outputs]).detach().mean()
        avg_train_rmse = torch.tensor([x['rmse_loss'] for x in outputs]).detach().mean()

        # Добавляем средние ошибки в словарь статистики обучения, используется для построение графиков
        self.train_history['train_loss'].append(avg_train_loss.numpy().item())
        self.train_history['train_rmse'].append(avg_train_rmse.numpy().item())

        # Если включено отображение графика обучения в конце эпохи, то рисуем графики
        if self.plot_epoch_loss:
            self.plot_history_loss()

    def validation_step(self, batch, batch_idx):
        """ Шаг валидации """
        return self._share_step(batch, batch_idx, mode='val')

    def validation_epoch_end(self, outputs):
        """Действия после окончания каждой эпохи валидации

        Параметры
        ---------
        outputs : list
          Список словарей.
          Каждый словарь - результат функции self._share_step для определенного батча на шаге валидации
        """

        # Считаем средние ошибки loss и rmse_loss по эпохе
        avg_val_loss = torch.tensor([x['loss'] for x in outputs]).detach().mean()
        avg_val_rmse = torch.tensor([x['rmse_loss'] for x in outputs]).detach().mean()
        # Логируем ошибку валидации
        self.log(f'val_loss', avg_val_loss, prog_bar=True)

        # Добавляем средние ошибки в словарь статистики обучения, используется для построение графиков
        self.train_history['val_loss'].append(avg_val_loss.numpy().item())
        self.train_history['val_rmse'].append(avg_val_rmse.numpy().item())

        # Если включено отображение графика обучения в конце эпохи, то рисуем графики
        if self.plot_epoch_loss:
            self.plot_history_loss()

    def plot_history_loss(self, clear_output=True):
        """ Функция построения графика обучения в конце эпохи

        Функция выводит 1 или 2 графика. Если включен режим без использования весов,
        то будет выведен один график с историей rmse на обучении и валидации.

        Если используется режим учета весов, то первый график будет показывать взвешенный RMSE на обучении
        и валидации, а второй обычный RMSE на обучении и валидации
        """

        if self.use_weights:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(7, 5))
            axes = [axes]

        axes[0].plot(np.arange(0, len(self.train_history['train_loss'])),
                     self.train_history['train_loss'],
                     label="train_loss")
        axes[0].scatter(np.arange(0, len(self.train_history['train_loss'])),
                     self.train_history['train_loss'])
        axes[0].plot(np.arange(0, len(self.train_history['val_loss'])),
                     self.train_history['val_loss'],
                     label="val_loss")
        axes[0].scatter(np.arange(0, len(self.train_history['val_loss'])),
                     self.train_history['val_loss'])
        axes[0].legend(loc='best')
        axes[0].set_xlabel("epochs")
        axes[0].set_ylabel("loss")
        val_loss_epoch_min = np.argmin(self.train_history['val_loss'])
        val_loss_min = self.train_history['val_loss'][val_loss_epoch_min]
        val_loss_min = round(val_loss_min, 3) if not np.isnan(val_loss_min) else val_loss_min
        title_min_vals = f'\nValidation minimum {val_loss_min} on epoch {val_loss_epoch_min}'
        if self.use_weights:
            axes[0].set_title('MODEL LOSS: Weighted RMSE'+title_min_vals)
        else:
            axes[0].set_title('MODEL LOSS: RMSE'+title_min_vals)
        axes[0].grid()

        if self.use_weights:
            axes[1].plot(np.arange(0, len(self.train_history['train_rmse'])),
                         self.train_history['train_rmse'], label="train_rmse")
            axes[1].scatter(np.arange(0, len(self.train_history['train_rmse'])),
                         self.train_history['train_rmse'])
            axes[1].plot(np.arange(0, len(self.train_history['val_rmse'])),
                         self.train_history['val_rmse'], label="val_rmse")
            axes[1].scatter(np.arange(0, len(self.train_history['val_rmse'])),
                         self.train_history['val_rmse'])
            axes[1].legend(loc='best')
            axes[1].set_xlabel("epochs")
            axes[1].set_ylabel("rmse")
            axes[1].set_title('MONITORING LOSS: RMSE without weights')
            axes[1].grid()

        plt.show()
        if clear_output:
            display.clear_output(wait=True)


class PetFinderTransferModelWithAddFeatures(PetFinderTransferModel):
    """ Модель на основе ошибки RMSE с использованием дополнительных признаков:
    'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
     'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur', 
      'n_pets', 'is_unknown', 'is_cat', 'is_dog', 'x_min', 'x_max', 'y_min', 'y_max', 'pet_ratio'
    """

    def __init__(self, model_name='resnet18', output_dims=None, dropout=0.5, learning_rate=0.01, full_trainable=False,
                 l2_regularization=1e-3, adam_betas=(0.9, 0.999), pretrained=True, use_weights=False,
                 plot_epoch_loss=True, seed=None):

        super().__init__(model_name=model_name, output_dims=output_dims, dropout=dropout, learning_rate=learning_rate,
                         full_trainable=full_trainable, l2_regularization=l2_regularization, adam_betas=adam_betas,
                         pretrained=pretrained, use_weights=use_weights, plot_epoch_loss=plot_epoch_loss, seed=seed)

    def _make_model_layers(self, model_name, output_dims, dropout, full_trainable, pretrained):

        model = TransferNet(output_dims=output_dims, dropout=dropout, full_trainable=full_trainable,
                            pretrained=pretrained)(name=model_name)

        """ В этом блоке выделим в отдельные переменные экстрактор признаков модели и полносвязную сеть.
        Так как может возникнуть задача добавления к признакам сверточной сети дополнительных признаков из вне.
        Так как меняется размер признаков, необходимо изменить размер входа полносвязной сети.
        В модели self.model размер входа полносвязной сети равен выходу стандартного экстрактора признаков
        """

        additive_features_names = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
                                   'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur', 
                                   'n_pets', 'is_unknown', 'is_cat', 'is_dog', 'x_min', 'x_max', 'y_min', 'y_max', 'pet_ratio']

        # В зависимости от модели полносвязный слой может называться по-разному
        # Находим имя полносвязной сети
        fc_name = self._get_model_fc_layer_name(model)

        # Когда нашли имя полносвязной сети, получаем параметры размера входа и выхода первого линейного слоя
        # Линейный слой может быть либо на первом месте, либо на втором. Перед ним может быть блок Dropout
        first_head_layer = getattr(model, fc_name)[0]
        if getattr(first_head_layer, 'in_features', None) is not None:
            head_first_linear_index = 0
        else:
            head_first_linear_index = 1

        in_features = getattr(model, fc_name)[head_first_linear_index].in_features
        out_features = getattr(model, fc_name)[head_first_linear_index].out_features

        # К текущему размеру входа прибавляем длину дополнительных параметров
        in_features += len(additive_features_names)

        # Копируем в отдельную переменную полносвязную сеть
        self.head_fc = getattr(model, fc_name)
        # И заменяем у нее первый линейный слой на такой же слой с увеличенным размером входа
        self.head_fc[head_first_linear_index] = nn.Linear(in_features=in_features, out_features=out_features)

        # А в исходной модели в полносвязный слой устанавливаем слой Identity.
        # Таким образом теперь self.model работает как экстрактор признаков из изображения
        setattr(model, fc_name, nn.Identity())
        self.feature_extractor = model

    def forward(self, x_in):
        # Извлекаем признаки из изображения
        image_features = self.feature_extractor(x_in['x_data'])
        # Соединяем признаки изображения с дополнительными признаками
        all_features = torch.hstack([image_features, x_in['additional_features']])
        # Получаем выход полносвязной сети
        x_out = self.head_fc(all_features).squeeze()
        return x_out


class PetFinderTransferModelBCE(PetFinderTransferModel):

    def __init__(self, model_name='resnet18', output_dims=None, dropout=0.5, learning_rate=0.01, full_trainable=False,
                 l2_regularization=1e-3, adam_betas=(0.9, 0.999), pretrained=True, use_weights=False,
                 use_mixup=False, mixup_alpha=0.4, mixup_prob=0.5, plot_epoch_loss=True, seed=None):

        super().__init__(model_name=model_name, output_dims=output_dims, dropout=dropout, learning_rate=learning_rate,
                         l2_regularization=l2_regularization, adam_betas=adam_betas, pretrained=pretrained,
                         use_weights=use_weights, plot_epoch_loss=plot_epoch_loss, full_trainable=full_trainable,
                         seed=seed)

        self._use_mixup = use_mixup
        self._mixup_alpha = mixup_alpha
        self._mixup_prob = mixup_prob

    def _mixup(self, images, target):
        assert self._mixup_alpha > 0, "alpha should be larger than 0"
        assert images.size(0) > 1, "Mixup cannot be applied to a single instance."

        lam = np.random.beta(self._mixup_alpha, self._mixup_alpha)
        rand_index = torch.randperm(images.size()[0])
        mixed_images = lam * images + (1 - lam) * images[rand_index, :]
        target_a, target_b = target, target[rand_index]

        return mixed_images, target_a, target_b, lam

    def forward(self, x_in, sigmoid=True, return_features=False):
        # Сделал для поддержки GradCam и Shap
        if isinstance(x_in, dict):
            image = x_in['x_data']
        else:
            image = x_in
        features = self.feature_extractor(image)
        x_out = self.head_fc(features).squeeze()
        if sigmoid:
            x_out = torch.sigmoid(x_out)
        if return_features:
            return x_out, features
        return x_out

    def _share_step(self, batch, batch_idx, mode='train'):
        """ Общий шаг для обучения, валидации и теста

        Параметры
        ---------
        batch : dict
          Батч-словарь в следующем формате:
          {
          'hash': список хэшей изображения,
          'x_data': список тензоров изображения размером N x C x H x W,
          'image_path': список путей к файлам изображений,
          'pawpularity': список метрик популярности изображения (может отсутствовать, см. описание класса PetDataset),
          }
        batch_idx : int
          Номер батча
        mode : str
          Режим. Используется только для префикса названий ошибок в логе.
          По умолчанию train
        """

        target_pawpularity = batch['pawpularity']

        if self._use_mixup and mode == 'train' and torch.rand(1)[0] <= self._mixup_prob:

            target_pawpularity_prob = target_pawpularity / 100

            mixed_images, target_a, target_b, lam = self._mixup(batch['x_data'], target_pawpularity_prob)

            batch['x_data'] = mixed_images

            # Получаем логиты для батча
            logits = self(batch, sigmoid=False)

            # Считаем ошибку ВСЕ и логируем ее
            bce_loss = lam * F.binary_cross_entropy_with_logits(logits, target_a, reduction='mean') + \
                       (1-lam) * F.binary_cross_entropy_with_logits(logits, target_b, reduction='mean')

            self.log(f'bce_{mode}_loss', bce_loss, prog_bar=True)

            pred_pawpularity = torch.sigmoid(logits) * 100
            rmse_loss = torch.sqrt(F.mse_loss(pred_pawpularity, target_pawpularity, reduction='mean'))

            self.log(f'rmse_{mode}_loss', rmse_loss, prog_bar=True)

            return {'loss': bce_loss, 'rmse_loss': rmse_loss.detach()}

        else:

            # Получаем предсказанную популярность для батча.
            logits = self(batch, sigmoid=False)

            # Считаем ошибку ВСЕ и логируем ее
            bce_loss = F.binary_cross_entropy_with_logits(logits, target_pawpularity / 100.0, reduction='mean')
            self.log(f'bce_{mode}_loss', bce_loss, prog_bar=True)

            pred_pawpularity = torch.sigmoid(logits) * 100
            rmse_loss = torch.sqrt(F.mse_loss(pred_pawpularity, target_pawpularity, reduction='mean'))
            self.log(f'rmse_{mode}_loss', rmse_loss, prog_bar=True)

            return {'loss': bce_loss, 'rmse_loss': rmse_loss.detach()}

    def plot_history_loss(self, clear_output=True):
        """ Функция построения графика обучения в конце эпохи

        Функция выводит 1 или 2 графика. Если включен режим без использования весов,
        то будет выведен один график с историей rmse на обучении и валидации.

        Если используется режим учета весов, то первый график будет показывать взвешенный RMSE на обучении
        и валидации, а второй обычный RMSE на обучении и валидации

        Параметры
        ---------
        clear_output : bool
          Испольщовать ли display.clear_output для очищения вывода ячейки.
          Используется при обучении в цикле для обновления графика.
        """

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(np.arange(0, len(self.train_history['train_loss'])),
                     self.train_history['train_loss'],
                     label="train_loss")
        axes[0].scatter(np.arange(0, len(self.train_history['train_loss'])),
                     self.train_history['train_loss'])
        axes[0].plot(np.arange(0, len(self.train_history['val_loss'])),
                     self.train_history['val_loss'],
                     label="val_loss")
        axes[0].scatter(np.arange(0, len(self.train_history['val_loss'])),
                     self.train_history['val_loss'])
        axes[0].legend(loc='best')
        axes[0].set_xlabel("epochs")
        axes[0].set_ylabel("loss")
        val_loss_epoch_min = np.argmin(self.train_history['val_loss'])
        val_loss_min = self.train_history['val_loss'][val_loss_epoch_min]
        val_loss_min = round(val_loss_min, 3) if not np.isnan(val_loss_min) else val_loss_min
        title_min_vals = f'\nValidation minimum {val_loss_min} on epoch {val_loss_epoch_min}'
        axes[0].set_title('MODEL LOSS: Cross Entropy'+title_min_vals)
        axes[0].grid()

        axes[1].plot(np.arange(0, len(self.train_history['train_rmse'])),
                     self.train_history['train_rmse'], label="train_rmse")
        axes[1].scatter(np.arange(0, len(self.train_history['train_rmse'])),
                        self.train_history['train_rmse'])
        axes[1].plot(np.arange(0, len(self.train_history['val_rmse'])),
                     self.train_history['val_rmse'], label="val_rmse")
        axes[1].scatter(np.arange(0, len(self.train_history['val_rmse'])),
                        self.train_history['val_rmse'])
        axes[1].legend(loc='best')
        axes[1].set_xlabel("epochs")
        axes[1].set_ylabel("rmse")
        val_rmse_epoch_min = np.argmin(self.train_history['val_rmse'])
        val_rmse_min = self.train_history['val_rmse'][val_rmse_epoch_min]
        val_rmse_min = round(val_rmse_min, 3) if not np.isnan(val_rmse_min) else val_rmse_min
        title_min_vals = f'\nValidation minimum {val_rmse_min} on epoch {val_rmse_epoch_min}'
        axes[1].set_title('MONITORING LOSS: RMSE'+title_min_vals)
        axes[1].grid()

        plt.show()

        if clear_output:
            display.clear_output(wait=True)


class PetFinderTransferModelBCEWithAddFeatures(PetFinderTransferModelBCE, PetFinderTransferModelWithAddFeatures):
    """ Модель на основе ошибки BCE с использованием дополнительных признаков:
    'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
    'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
    """

    def __init__(self, model_name='resnet18', output_dims=None, dropout=0.5, learning_rate=0.01, full_trainable=False,
                 l2_regularization=1e-3, adam_betas=(0.9, 0.999), pretrained=True, use_weights=False,
                 use_mixup=False, mixup_alpha=0.4, mixup_prob=0.5, plot_epoch_loss=True, seed=None):

        PetFinderTransferModelBCE.__init__(self, model_name=model_name, output_dims=output_dims, dropout=dropout,
                                           learning_rate=learning_rate,
                                           full_trainable=full_trainable,
                                           l2_regularization=l2_regularization, adam_betas=adam_betas,
                                           use_mixup=use_mixup, mixup_alpha=mixup_alpha,
                                           mixup_prob=mixup_prob,
                                           pretrained=pretrained, use_weights=use_weights,
                                           plot_epoch_loss=plot_epoch_loss, seed=seed)

        PetFinderTransferModelWithAddFeatures.__init__(self, model_name=model_name, output_dims=output_dims,
                                                       dropout=dropout, learning_rate=learning_rate,
                                                       full_trainable=full_trainable,
                                                       l2_regularization=l2_regularization,
                                                       adam_betas=adam_betas,
                                                       pretrained=pretrained, use_weights=use_weights,
                                                       plot_epoch_loss=plot_epoch_loss, seed=seed)

    def forward(self, x_in, sigmoid=True, return_features=False):
        # Извлекаем признаки из изображения
        image_features = self.feature_extractor(x_in['x_data'])
        # Соединяем признаки изображения с дополнительными признаками
        all_features = torch.hstack([image_features, x_in['additional_features']])
        # Получаем выход полносвязной сети
        x_out = self.head_fc(all_features).squeeze()
        if sigmoid:
            x_out = torch.sigmoid(x_out)
        if return_features:
            return x_out, all_features
        return x_out


class PetFinderTransferModelKL(PetFinderTransferModel):
    """ Модель для трансферного обучения с помощью ошибки RMSE + KLDiv

    Функция ошибок:
    loss = KLDiVLoss + rmse_part_coef * RMSE

    Последний слой полносвязной сети имеет 100 выходов. Предсказанное значение популярности для RMSE находится
    с помощью argmax. В функции ошибок KLDiVLoss вычисляется похожесть предсказанного распределения с помощью
    Softmax. Сравниваются предсказанное респределение вероятностей и целевого распределения по формуле Гаусса.
    RMSE будет всегда значительно больше ошибки KLDiVLoss, это нужно учитывать при подборе коэффициента rmse_part_coef


    Параметры
    ---------
    model_name : str
      Название модели. Может принимать значение resnet18, vgg16, alexnet или googlenet
      По умолчанию resnet18.
    rmse_part_coef : float
      Коэффициент учета ошибки RMSE в функции общей ошибки

    output_dims : list
      Структура полносвязной сети на выходе экстратора признаков.
      Длина списка - количество полносвязных слоев.
      Значения списка - кол-во нейронов в каждом полносвязном слое.
      Пример: [1024, 512, 256, 100] - четыре полносвязных слоя.
              Первый слой с 1024 нейронами, второй слой с 512 нейронами,
              третий слой с 256 нейронами и последний слой с 100 нейроном.
      По умолчанию [128, 256, 100].
      !!! Последний слой должен обязательно иметь значение 100
    dropout : float
      Значение вероятности для всех слоев Dropout полносвязной сети.
      По умолчанию 0.5
    learning_rate : float
      Скорость обучения. По умолчанию 0.01
    l2_regularization : float
      Регуляризация весов L2. По умолчанию 0.001
    adam_betas : tuple
      Коэффициенты оптимизатора Adam. По умолчанию (0.9, 0.999)
    pretrained : bool
      Если True, то будут загружена предобученная модель на ImageNet.
      По умолчанию True.
      !!! ВАЖНО: Класс не тестировался на не предобученных моделях.
          Структура модели при значении pretrained=False может отличаться, поведение класса может быть некорректным.
    use_weights : bool
      Флаг использования весов классов в функции ошибок. По умолчанию False.
    plot_epoch_loss : bool
      Флаг печати графиков ошибок в конце каждой эпохи во время обучения. По умолчанию True.
    seed : int
      Фиксация генератора случайных чисел.
    """

    def __init__(self, model_name='resnet18', output_dims=None, dropout=0.5, learning_rate=0.01, full_trainable=False,
                 l2_regularization=1e-3, adam_betas=(0.9, 0.999), pretrained=True, l1_part_coef=1.0,
                 use_weights=False, plot_epoch_loss=True, seed=None):

        if not output_dims:
            output_dims = [128, 256, 100]

        super().__init__(model_name=model_name, output_dims=output_dims, dropout=dropout, learning_rate=learning_rate,
                         l2_regularization=l2_regularization, adam_betas=adam_betas, pretrained=pretrained,
                         use_weights=use_weights, plot_epoch_loss=plot_epoch_loss, full_trainable=full_trainable,
                         seed=seed)

        self.train_history = {
            'train_loss': [np.nan],
            'train_kldiv': [np.nan],
            'train_l1': [np.nan],
            'train_rmse': [np.nan],
            'val_loss': [],
            'val_kldiv': [],
            'val_l1': [],
            'val_rmse': [],
        }

        self._l1_part_coef = l1_part_coef
        self.save_hyperparameters()

    @staticmethod
    def _check_output_dims(output_dims):
        assert output_dims[-1] == 100, f"Кол-во нейронов в последнем слое output_dims должен быть равен 100. Текущее значение: {output_dims}"

    @staticmethod
    def _kldiv_loss(prediction, target, weights=None):
        """ Функция расчета ошибки KLDiv

        Параметры
        ---------
        prediction : array
          Список предсказанных векторов
        target : список целевых векторов

        Результат
        ---------
        kldivloss : float
        """

        # Прибавляются 1e-9 в числителе и знаменателе чтобы избежать значений бесконечности
        # Суммируем ошибки только по строкам. Такой выход потребуется, если будет необходимо применить веса
        # для каждого экземпляра ошибки
        loss = torch.sum(target * torch.log((target + 1e-9) / (prediction + 1e-9)), dim=1)
        #loss = torch.sum(target * torch.log(prediction+1e-9), dim=1)

        # Иногда функция ошибок может дать результат +inf, поэтому ограничиваем значение ошибвки верхним порогом в 1e9
        # Так же функция не может принимать значение меньше нуля
        loss = torch.clamp(loss, 0, 1e9)

        if weights is not None:
            loss = torch.sum(loss * weights) / len(loss)
        else:
            loss = torch.sum(loss) / len(loss)

        return loss
    
    @staticmethod
    def _l1_loss(prediction, pawpularity, weights=None):
        """ Функция расчета ошибки KLDiv

        Параметры
        ---------
        prediction : array
          Список предсказанных векторов
        target : список целевых векторов

        Результат
        ---------
        kldivloss : float
        """
        
        l1_vec = torch.vstack([torch.arange(0, 100)[None, :]]*prediction.shape[0])
        l1_vec = l1_vec.to(prediction.device)

        # Прибавляются 1e-9 в числителе и знаменателе чтобы избежать значений бесконечности
        # Суммируем ошибки только по строкам. Такой выход потребуется, если будет необходимо применить веса
        # для каждого экземпляра ошибки
        loss = torch.abs(torch.sum(prediction*l1_vec)+1 - pawpularity)

        # Иногда функция ошибок может дать результат +inf, поэтому ограничиваем значение ошибвки верхним порогом в 1e9
        # Так же функция не может принимать значение меньше нуля
        loss = torch.clamp(loss, 0, 1e9)

        if weights is not None:
            loss = torch.sum(loss * weights) / len(loss)
        else:
            loss = torch.sum(loss) / len(loss)

        return loss
    

    def _share_step(self, batch, batch_idx, mode='train'):
        """ Общий шаг для обучения, валидации и теста

        Параметры
        ---------
        batch : dict
          Батч-словарь в следующем формате:
          {
          'hash': список хэшей изображения,
          'x_data': список тензоров изображения размером N x C x H x W,
          'image_path': список путей к файлам изображений,
          'pawpularity': список метрик популярности изображения (может отсутствовать, см. описание класса PetDataset),
          'class_weight': список списков  весов класса изображения (может отсутствовать, см. описание класса PetDataset),
          'label': список меткок в формате вектора в виде нормального распределения со средним в значении метки класса
                   (может отсутствовать, см. описание класса PetDataset)
          }
        batch_idx : int
          Номер батча
        mode : str
          Режим. Используется только для префикса названий ошибок в логе.
          По умолчанию train
        """

        target_label = batch['label_distribution']
        weights = batch.get('class_weight', None)
        target_pawpularity = batch['pawpularity']

        # Получаем вектор предсказания модели
        y_pred = self(batch)
        # Применяем к вектору предсказания Softmax, требуется для вычисления ошибки KLDiv,
        # в ней используются вероятности
        y_pred_softmax = F.softmax(y_pred, dim=1)
        
        # Получаем предсказание популярности для расчета RMSE.
        pred_pawpularity = torch.argmax(y_pred_softmax, dim=1)+1
        # Вычисляем ошибку RMSE без учета весов и логируем ее
        rmse_loss = self._rmse_loss(predict=pred_pawpularity, target=target_pawpularity)
        self.log(f'rmse_{mode}_loss', rmse_loss, prog_bar=True)        

        # Вычисляем ошибку L1 без учета весов и логируем ее
        l1_loss = self._l1_loss(prediction=y_pred_softmax, pawpularity=target_pawpularity)
        self.log(f'l1_{mode}_loss', l1_loss, prog_bar=True)

        # Находим среднюю по батчу ошибку KLDiv без учета весов и логируем ее
        kldiv_loss = self._kldiv_loss(prediction=y_pred_softmax, target=target_label)
        self.log(f'kldiv_{mode}_loss', kldiv_loss, prog_bar=True)

        # Блок расчета оптимизируемой ошибки
        if self.use_weights:
            # Если включен режим весов, то находим взвешенную ошибку KLDiv по батчку
            kldiv_weighted = self._kldiv_loss(prediction=y_pred_softmax, target=target_label, weights=weights)
            # Пересчитываетм L1 с учетом весов
            l1_loss_weighted = self._l1_loss(prediction=y_pred_softmax, pawpularity=target_pawpularity,
                                             weights=weights)
            # Общая ошибка равна взвешенной KLDiv + взвешенная L1 с коэффициентом учета
            loss = kldiv_weighted + self._l1_part_coef * l1_loss_weighted
        else:
            # Если режим весов выключен, то общая ошибка равна KLDiv + L1 с коэффициентом учета
            loss = kldiv_loss + self._l1_part_coef * l1_loss

        # Логируем общую ошибку, а так же невзвешенные KLDiv, L1 и RMSE
        return {'loss': loss, 'kldiv_loss': kldiv_loss.detach(), 'l1_loss': l1_loss.detach(), 'rmse_loss': l1_loss.detach()}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        self.log(f'val_loss', avg_val_loss, prog_bar=True)

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.tensor([x['loss'] for x in outputs]).detach().mean()
        avg_train_kldiv = torch.tensor([x['kldiv_loss'] for x in outputs]).detach().mean()
        avg_train_l1 = torch.tensor([x['l1_loss'] for x in outputs]).detach().mean()
        avg_train_rmse = torch.tensor([x['rmse_loss'] for x in outputs]).detach().mean()

        self.train_history['train_loss'].append(avg_train_loss.numpy().item())
        self.train_history['train_kldiv'].append(avg_train_kldiv.numpy().item())
        self.train_history['train_l1'].append(avg_train_l1.numpy().item())
        self.train_history['train_rmse'].append(avg_train_rmse.numpy().item())

        if self.plot_epoch_loss:
            self.plot_history_loss()

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in outputs]).detach().mean()
        avg_val_kldiv = torch.tensor([x['kldiv_loss'] for x in outputs]).detach().mean()
        avg_val_l1 = torch.tensor([x['l1_loss'] for x in outputs]).detach().mean()
        avg_val_rmse = torch.tensor([x['rmse_loss'] for x in outputs]).detach().mean()
        self.log(f'val_loss', avg_val_loss, prog_bar=True)

        self.train_history['val_loss'].append(avg_val_loss.numpy().item())
        self.train_history['val_kldiv'].append(avg_val_kldiv.numpy().item())
        self.train_history['val_l1'].append(avg_val_l1.numpy().item())
        self.train_history['val_rmse'].append(avg_val_rmse.numpy().item())

        # Если включено отображение графика обучения в конце эпохи, то рисуем графики
        if self.plot_epoch_loss:
            self.plot_history_loss()

    def plot_history_loss(self, clear_output=True):
        """ Функция построения графика обучения в конце эпохи

        Функция выводит 3 графика.

        Первый показывает ошибку RMSE + KLDiv на обучении и валидации. Второй график показывает только ошибку
        KLDiv на обучении и валидации, а третий ошибку RMSE на обучении и валидации.

        Так же если включен режим учета весов, то первый график будет показывать суммы взвешенных KLDiv и RMSE.
        Остальные графики показывают тоже самое, веса на них не учитываются.
        """

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))

        axes[0].plot(np.arange(0, len(self.train_history['train_loss'])),
                     self.train_history['train_loss'],
                     label="train_loss")
        axes[0].scatter(np.arange(0, len(self.train_history['train_loss'])),
                        self.train_history['train_loss'])
        axes[0].plot(np.arange(0, len(self.train_history['val_loss'])),
                     self.train_history['val_loss'],
                     label="val_loss")
        axes[0].scatter(np.arange(0, len(self.train_history['val_loss'])),
                       self.train_history['val_loss'])
        axes[0].legend(loc='best')
        axes[0].set_xlabel("epochs")
        axes[0].set_ylabel("loss")
        val_loss_epoch_min = np.argmin(self.train_history['val_loss'])
        val_loss_min = self.train_history['val_loss'][val_loss_epoch_min]
        val_loss_min = round(val_loss_min, 3) if not np.isnan(val_loss_min) else val_loss_min
        title_min_vals = f'\nValidation minimum {val_loss_min} on epoch {val_loss_epoch_min}'
        if self.use_weights:
            axes[0].set_title(f'MODEL LOSS: Weighted KLDiv + Weighted L1\nL1 Coef: {self._l1_part_coef}'+title_min_vals)
        else:
            axes[0].set_title(f'MODEL LOSS: KLDiv + L1\nL1 Coef: {self._l1_part_coef}'+title_min_vals)
        axes[0].grid()

        axes[1].plot(np.arange(0, len(self.train_history['train_rmse'])),
                     self.train_history['train_rmse'],
                     label="train_rmse")
        axes[1].scatter(np.arange(0, len(self.train_history['train_rmse'])),
                        self.train_history['train_rmse'])
        axes[1].plot(np.arange(0, len(self.train_history['val_rmse'])),
                     self.train_history['val_rmse'],
                     label="val_rmse")
        axes[1].scatter(np.arange(0, len(self.train_history['val_rmse'])),
                        self.train_history['val_rmse'])
        axes[1].legend(loc='best')
        axes[1].set_xlabel("epochs")
        axes[1].set_ylabel("rmse")
        val_rmse_epoch_min = np.argmin(self.train_history['val_rmse'])
        val_rmse_min = self.train_history['val_rmse'][val_rmse_epoch_min]
        val_rmse_min = round(val_rmse_min, 3) if not np.isnan(val_rmse_min) else val_rmse_min
        title_min_vals = f'\nValidation minimum {val_rmse_min} on epoch {val_rmse_epoch_min}'
        axes[1].set_title('MONITORING LOSS: RMSE'+title_min_vals)
        axes[1].grid()

        axes[2].plot(np.arange(0, len(self.train_history['train_kldiv'])),
                     self.train_history['train_kldiv'],
                     label="train_kldiv")
        axes[2].scatter(np.arange(0, len(self.train_history['train_kldiv'])),
                        self.train_history['train_kldiv'])
        axes[2].plot(np.arange(0, len(self.train_history['val_kldiv'])),
                     self.train_history['val_kldiv'],
                     label="val_kldiv")
        axes[2].scatter(np.arange(0, len(self.train_history['val_kldiv'])),
                        self.train_history['val_kldiv'])
        axes[2].legend(loc='best')
        axes[2].set_xlabel("epochs")
        axes[2].set_ylabel("kldiv")
        val_kldiv_epoch_min = np.argmin(self.train_history['val_kldiv'])
        val_kldiv_min = self.train_history['val_kldiv'][val_kldiv_epoch_min]
        val_kldiv_min = round(val_kldiv_min, 3) if not np.isnan(val_kldiv_min) else val_kldiv_min
        title_min_vals = f'\nValidation minimum {val_kldiv_min} on epoch {val_kldiv_epoch_min}'
        axes[2].set_title('MONITORING LOSS: KLDiV'+title_min_vals)
        axes[2].grid()

        plt.show()

        if clear_output:
            display.clear_output(wait=True)


class PetFinderTransferModelKLWithAddFeatures(PetFinderTransferModelKL, PetFinderTransferModelWithAddFeatures):
    
    """ Модель для трансферного обучения с помощью ошибки RMSE + KLDiv c дополнительными признаками

    Функция ошибок:
    loss = KLDiVLoss + l1_part_coef * L1

    Последний слой полносвязной сети имеет 100 выходов. Предсказанное значение популярности для RMSE находится
    с помощью argmax. В функции ошибок KLDiVLoss вычисляется похожесть предсказанного распределения с помощью
    Softmax. Сравниваются предсказанное респределение вероятностей и целевого распределения по формуле Гаусса.
    RMSE будет всегда значительно больше ошибки KLDiVLoss, это нужно учитывать при подборе коэффициента rmse_part_coef


    Параметры
    ---------
    model_name : str
      Название модели. Может принимать значение resnet18, vgg16, alexnet или googlenet
      По умолчанию resnet18.
    l1_part_coef : float
      Коэффициент учета ошибки L1 в функции общей ошибки
    output_dims : list
      Структура полносвязной сети на выходе экстратора признаков.
      Длина списка - количество полносвязных слоев.
      Значения списка - кол-во нейронов в каждом полносвязном слое.
      Пример: [1024, 512, 256, 100] - четыре полносвязных слоя.
              Первый слой с 1024 нейронами, второй слой с 512 нейронами,
              третий слой с 256 нейронами и последний слой с 100 нейроном.
      По умолчанию [128, 256, 100].
      !!! Последний слой должен обязательно иметь значение 100
    dropout : float
      Значение вероятности для всех слоев Dropout полносвязной сети.
      По умолчанию 0.5
    learning_rate : float
      Скорость обучения. По умолчанию 0.01
    l2_regularization : float
      Регуляризация весов L2. По умолчанию 0.001
    adam_betas : tuple
      Коэффициенты оптимизатора Adam. По умолчанию (0.9, 0.999)
    pretrained : bool
      Если True, то будут загружена предобученная модель на ImageNet.
      По умолчанию True.
      !!! ВАЖНО: Класс не тестировался на не предобученных моделях.
          Структура модели при значении pretrained=False может отличаться, поведение класса может быть некорректным.
    use_weights : bool
      Флаг использования весов классов в функции ошибок. По умолчанию False.
    plot_epoch_loss : bool
      Флаг печати графиков ошибок в конце каждой эпохи во время обучения. По умолчанию True.
    seed : int
      Фиксация генератора случайных чисел.
    """

    def __init__(self, model_name='resnet18', output_dims=None, dropout=0.5, learning_rate=0.01, full_trainable=False,
                 l2_regularization=1e-3, adam_betas=(0.9, 0.999), pretrained=True, l1_part_coef=1.0,
                 use_weights=False, plot_epoch_loss=True, seed=None):

        PetFinderTransferModelKL.__init__(self, model_name=model_name, output_dims=output_dims, dropout=dropout, learning_rate=learning_rate, 
                                          full_trainable=full_trainable, l2_regularization=l2_regularization, adam_betas=adam_betas, 
                                          pretrained=pretrained, l1_part_coef=l1_part_coef, use_weights=use_weights, plot_epoch_loss=plot_epoch_loss, 
                                          seed=seed)

        PetFinderTransferModelWithAddFeatures.__init__(self, model_name=model_name, output_dims=output_dims,
                                                       dropout=dropout, learning_rate=learning_rate,
                                                       full_trainable=full_trainable,
                                                       l2_regularization=l2_regularization,
                                                       adam_betas=adam_betas,
                                                       pretrained=pretrained, use_weights=use_weights,
                                                       plot_epoch_loss=plot_epoch_loss, seed=seed)
        
        self.train_history = {
            'train_loss': [np.nan],
            'train_kldiv': [np.nan],
            'train_l1': [np.nan],
            'train_rmse': [np.nan],
            'val_loss': [],
            'val_kldiv': [],
            'val_l1': [],
            'val_rmse': [],
        } 
          
    def forward(self, x_in):
        # Извлекаем признаки из изображения
        image_features = self.feature_extractor(x_in['x_data'])
        # Соединяем признаки изображения с дополнительными признаками
        all_features = torch.hstack([image_features, x_in['additional_features']])
        # Получаем выход полносвязной сети
        x_out = self.head_fc(all_features).squeeze()
        return x_out
        
        
    
    
