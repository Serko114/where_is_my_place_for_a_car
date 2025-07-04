from ultralytics import YOLO
import torch
import numpy as np
import cv2

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
# from byte_tracker.byte_tracker_model import BYTETracker as ByteTracker


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
# ---------------------------------------------код-------------------------------------------------------------------
        # '-'
        # '-'
        # '-'
        # '-'
        # ENCODER = 'resnet18'
        # ENCODER_WEIGHTS = 'imagenet'
        # preprocessing_fn = smp.encoders.get_preprocessing_fn(
        #     self.encoder, self.encoder_weights)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # preprocessing_fn = smp.encoders.get_preprocessing_fn(
        #     'resnet18', 'imagenet')
        dataset = Dataset(frame, 'utils_local/костыль.png', augmentation=gva(), preprocessing=get_preprocessing(self.preprocessing_fn)
                          )
        # test_dataset = Dataset(frame, 'utils_local/костыль.png', augmentation=gva,
        #                        preprocessing=get_preprocessing(preprocessing_fn))
        indx = np.random.randint(len(dataset))
        image, mask_gt = dataset[indx]

        # visualize_multichennel_mask(image, mask)
        print(f'это кратинка{image.shape, image.dtype}: {image}')
        # .transpose(2, 0, 1).astype('float32')
        # x_tensor = image.transpose(2, 0, 1).astype('float32')
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        print(
            f'это кратинка для модели{x_tensor.shape, image.dtype}: {x_tensor}')
        # '-'
        # '-'
        # '-'
        # '-'
        pr_mask = self.best_model(x_tensor)
        pr_mask = pr_mask.squeeze().cpu().detach().numpy()
        label_mask = np.argmax(pr_mask, axis=0)
        print(label_mask.shape, image.shape, mask_gt.shape)
        print(
            f'это МАСКА: {label_mask.shape, label_mask.dtype}: {label_mask}')
        real, pre = visualize_predicts(image, np.argmax(
            mask_gt, axis=0), label_mask, normalized=True)

        # '-'
        # '-'
        # '-'
        # '-'
        # for i in test_dataset:
        #     print(test_dataset)
        # '-'

        #     DEVICE = torch.device(
        # "cuda" if torch.cuda.is_available() else "cpu")
        # image = frame_element.frame
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # transform = gva()
        # transformed = transform(image=image)
        # print(
        # f'-------------------{transformed['image'].shape}-----------------------------')
        # pica = transformed['image']
        # pic = pica.transpose(2, 0, 1).astype('float32')
        # x_tensor = torch.from_numpy(pic).to(DEVICE).unsqueeze(0)
        # mask = self.best_model(x_tensor)
        # frame_element.frame_pr_mask = mask
        #  print(
        #  f'-------------------{mask.size()}-----------------------------')
        # ---смотрим---
        # cv2.imshow('dsd', image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # cv2.waitKey(1)
        # frame_pr_mask = frame_element.frame_pr_mask.cpu()
        #       print(
        #   f'-------------------{np.array(mask.detach().cpu()).shape}-----------------------------')
        #         show_mask = np.array(mask.detach().cpu())[0].transpose(1, 2, 0)
        #    print(
        #    f'-------------------{show_mask.shape}-----------------------------')

        # frame_pr_mask = frame_element.frame_pr_mask.cpu
        # mask = np.array(frame_pr_mask)
        # print(f'маска: {mask.shape}')
        # print(np.array(frame_pr_mask).shape)
        # ------------------------------------------------------cv2.imshow('Webcam', show_mask)
        # Выход из цикла по нажатию клавиши 'q'
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # ------------------------------------------------------cv2.waitKey(1)
        # -------------
        # print(frame_element.frame_pr_mask)
        return frame_element
        # ----------------------------------------------------------------------------------------------------------------
        #     outputs = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False,
        #                                  iou=self.iou, classes=self.classes_to_detect)
        #     # print(f'OUTPUTS: {outputs}')
        #     frame_element.detected_conf = outputs[0].boxes.conf.cpu().tolist()
        #     detected_cls = outputs[0].boxes.cls.cpu().int().tolist()
        #     frame_element.detected_cls = [self.classes[i] for i in detected_cls]
        #     frame_element.detected_xyxy = outputs[0].boxes.xyxy.cpu(
        #     ).int().tolist()

        #     # Преподготовка данных на подачу в трекер
        #     detections_list = self._get_results_dor_tracker(outputs)
        #     # на выходе будет вот это: [[277.84, 68.415, 429.72, 289.89, 0.98837, 0]]
        #     # print(f'РЕЗУЛЬТИРУЮЩИЙ СПИСОК: {detections_list}')
        #     # Если детекций нет, то оправляем пустой массив
        #     if len(detections_list) == 0:
        #         detections_list = np.empty((0, 6))

        #     track_list = self.tracker.update(
        #         torch.tensor(detections_list), xyxy=True)
        #     # print(f'ID: {[int(t.track_id) for t in track_list]}')
        #     # Получение id list
        #     frame_element.id_list = [int(t.track_id) for t in track_list]

        #     # Получение box list
        #     frame_element.tracked_xyxy = [
        #         list(t.tlbr.astype(int)) for t in track_list]

        #     # Получение object class names
        #     frame_element.tracked_cls = [
        #         self.classes[int(t.class_name)] for t in track_list]

        #     # Получение conf scores
        #     frame_element.tracked_conf = [t.score for t in track_list]
        #     # -------------------------------------------------------------
        #     # создаем список с указанием единичкой на класс [0 1 0 0 0 0] (словарь см. в config.yaml)
        #     frame_element.cls_id = list(self.f(track_list))
        #     # print(self.f(track_list))
        #     # -------------------------------------------------------------
        #     print(f'id: {[int(t.track_id) for t in track_list]} \
        #             рамка: {[list(t.tlbr.astype(int)) for t in track_list]} \
        #             класс: {[self.classes[int(t.class_name)] for t in track_list]} \
        #             бинарный класс:{list(self.f(track_list))} \
        #             вероятность: {[t.score for t in track_list]}')
        #     # типа результат: [1] [[272, 63, 424, 289]] ['joyful'] [0 1 0 0 0 0] [0.9907557368278503]
        #     # -------------------------------------------------------------
        #     return frame_element

        # def f(self, tr_lst) -> list:
        #     # функция для создания списка для подачи в столбцы эмоций в двоичном виде [0 1 0 0 0 0] (словарь см. в config.yaml)
        #     massive = np.array([int(i) for i in np.zeros(6)])
        #     classs_id = [self.classes[int(t.class_name)] for t in tr_lst][0]
        #     s = self.dict_clsses[classs_id]
        #     massive[s] = 1
        #     return massive

        # def _get_results_dor_tracker(self, results) -> np.ndarray:
        #     # Приведение данных в правильную форму для трекера
        #     detections_list = []
        #     for result in results[0]:
        #         class_id = result.boxes.cls.cpu().numpy().astype(int)
        #         # трекаем те же классы что и детектируем
        #         if class_id[0] in self.classes_to_detect:

        #             bbox = result.boxes.xyxy.cpu().numpy()
        #             confidence = result.boxes.conf.cpu().numpy()

        #             class_id_value = class_id[0]
        #             # class_id_value = (
        #             #     2  # Будем все трекуемые объекты считать классом car чтобы не было ошибок
        #             # )
        #             # print(confidence)
        #             # print(f'класс: {class_id_value} уверенность: {confidence[0]}')

        #             merged_detection = [
        #                 bbox[0][0],
        #                 bbox[0][1],
        #                 bbox[0][2],
        #                 bbox[0][3],
        #                 confidence[0],
        #                 class_id_value,
        #             ]
        #             # print(f'коробки: {merged_detection}')
        #             detections_list.append(merged_detection)
        #             # print(f'РЕЗУЛЬТАТ: {merged_detection}')
        #     # ----------------------------
        #     # отбирает одну детакцию с максимальным значением уверенности для одного id треккинга
        #     s = max([i[-2] for i in detections_list])
        #     detections_list = [i for i in detections_list if i[-2] == s]
        #     # ----------------------------
        #     # print(f'РЕЗУЛЬТАТ ОКОНЧАТЕЛЬНЫЙ: {detections_list}')
        #     return np.array(detections_list)
