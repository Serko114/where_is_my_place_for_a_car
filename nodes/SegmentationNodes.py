from ultralytics import YOLO
import torch
import numpy as np
import cv2
import pandas as pd

# from utils_local.utils import profile_time
from utils_local.utils import get_validation_augmentation as gva
# from utils_local.utils import get_validation_augmentation as gva
from utils_local.utils import visualize_multichennel_mask
from utils_local.utils import get_preprocessing
from utils_local.utils import visualize_predicts

from utils_local.Datasets import Dataset

from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import VideoEndBreakElement
import segmentation_models_pytorch as smp
import albumentations as albu


class SegmentationNodes:
    """Модуль инференса модели детекции + трекинг алгоритма"""

    def __init__(self, config) -> None:
        DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f'Детекция будет производиться на {DEVICE}')

        config_yolo = config["segmentation_node"]
        self.colors_imshow = config_yolo["colors_imshow"]
        # self.encoder = config_yolo['ENCODER']
        # self.encoder_weights = config_yolo['ENCODER_WEIGHTS']
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            'resnet18', 'imagenet')
        self.best_model = torch.jit.load(
            'models/best_segmentation.pt', map_location=DEVICE)

    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    def process(self, frame_element: FrameElement) -> FrameElement:
        # Выйти из обработки если это пришел VideoEndBreakElement а не FrameElement
        if isinstance(frame_element, VideoEndBreakElement):
            return frame_element
        assert isinstance(
            frame_element, FrameElement
        ), f"DetectionTrackingNodes | Неправильный формат входного элемента {type(frame_element)}"

        frame = frame_element.frame.copy()
# ---------------------------------------------код-------------------------------------------------------
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = Dataset(frame, 'utils_local/костыль.png', augmentation=gva(), preprocessing=get_preprocessing(self.preprocessing_fn)
                          )
        indx = np.random.randint(len(dataset))
        image, mask_gt = dataset[indx]

        # print(f'это кратинка{image.shape, image.dtype}: {image}')
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        # print(
        #     f'это кратинка для модели{x_tensor.shape, image.dtype}: {x_tensor}')
        pr_mask = self.best_model(x_tensor)  # предсказание маски
        # изменение формы и типа данных матрицы:
        pr_mask = pr_mask.squeeze().cpu().detach().numpy()
        # строка приведения каждого пикселя с значениям 0,1,2:
        label_mask = np.argmax(pr_mask, axis=0)
# -----------------------------------------COEFF---------------------------------------------------------
        # 2 - это парковка, 1 - это машины
        n = 0
        m = 0
        for i in label_mask:
            for j in i:
                if j == 1:
                    n += 1
                elif j == 2:
                    m += 1
        coeff = round(n / m, 4)
        frame_element.coeff = coeff
        print(f'coeff {frame_element.frame_num} : {coeff}')
# -------------------------------------------------------------------------------------------------------
        # print(label_mask.shape, image.shape, mask_gt.shape)
        # print(
        #     f'это МАСКА: {label_mask.shape, label_mask.dtype}: {label_mask}')
# -----------------------------------------смотрим_картинку----------------------------------------------
        # real, pre = visualize_predicts(image, np.argmax(
        #     mask_gt, axis=0), label_mask, normalized=True)
# -------------------------------------------------------------------------------------------------------
        return frame_element
