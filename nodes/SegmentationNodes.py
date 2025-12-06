from ultralytics import YOLO
import torch
import numpy as np
import cv2
import pandas as pd
from PIL import Image

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

from torch_snippets import read, show


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

# -----------------------------------------смотрим_картинку----------------------------------------------
        # print(f'Маааааааааааааааааааааааааааааааааааааска{label_mask.shape}')
        # print(f'Карррррррррррррррррррррррррррррррррртинка{frame.shape}')
        # lab_resize = label_mask.resize((1080, 1920))
        # im_mask = label_mask[56:200, :]  # .crop(0, 200, 256, 56)
        # res_img = cv2.resize(im_mask, (1080, 1920), cv2.INTER_NEAREST)
        # im = Image.fromarray(im_mask)
        # show(im_mask)
        # res_im = cv2.resize(im, (1080, 1920), cv2.INTER_NEAREST)
        # print(f'Маааааааааааааааааааааааааааааааааааааска{res_im.shape}')
        # show(res_im)
        # frame_ = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # cv2.imshow('Webcam', frame_)
        # cv2.imshow('Webcam', label_mask)
        # Выход из цикла по нажатию клавиши 'q'
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        cv2.waitKey(2)

        # show(label_mask)
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
# смотрим картинки и маски
# #Lets plot some samples
# rows,cols=3,3
# # fig=plt.figure(figsize=(10,10))
# for i in range(1,rows*cols+1):
#     fig.add_subplot(rows,cols,i)
#     img_path=image[i]
#     msk_path=mask[i]
#     img=cv2.imread(img_path)
#     img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     msk=cv2.imread(msk_path)
#     plt.imshow(img)
#     plt.imshow(msk,alpha=0.5)
#     plt.xticks([]), plt.yticks([])
# plt.show()
        return frame_element
