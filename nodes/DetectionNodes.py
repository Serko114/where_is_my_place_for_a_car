from ultralytics import YOLO
import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


# from utils_local.utils import profile_time
from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import VideoEndBreakElement
# from byte_tracker.byte_tracker_model import BYTETracker as ByteTracker


class DetectionNodes:
    """Модуль инференса модели детекции + трекинг алгоритма"""

    def __init__(self, config) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Детекция будет производиться на {device}')

        config_yolo = config["detection_node"]
        # YOLO(config_yolo["weight_pth"], task='detect')
        self.weight_pth = config_yolo['weight_pth']
        # self.classes = self.model.names
        self.conf = config_yolo["conf"]
        self.iou = config_yolo["iou"]
        self.imgsz = config_yolo["imgsz"]
        self.classes_to_detect = config_yolo["classes_to_detect"]
        self.dict_clsses = config_yolo["dict_clsses"]

    # @profile_time
    def process(self, frame_element: FrameElement) -> FrameElement:
        # Выйти из обработки если это пришел VideoEndBreakElement а не FrameElement
        if isinstance(frame_element, VideoEndBreakElement):
            return frame_element
        assert isinstance(
            frame_element, FrameElement
        ), f"DetectionTrackingNodes | Неправильный формат входного элемента {type(frame_element)}"

        frame = frame_element.frame.copy()
        model = YOLO(self.weight_pth)
        outputs = model(frame, imgsz=int(self.imgsz),
                        iou=float(self.iou), conf=float(self.conf), verbose=True,)
        #  iou=self.iou, classes=self.classes_to_detect)
        # print(f'OUTPUTS: {outputs}')
        frame_element.detected_conf = outputs[0].obb.conf.cpu().tolist()
        frame_element.detected_cls = outputs[0].obb.cls.cpu().int().tolist()
        # frame_element.detected_cls = [self.classes[i] for i in detected_cls]
        frame_element.detected_xyxy = outputs[0].obb.xyxyxyxy.cpu(
        ).int().tolist()
        self.detect_count = len(outputs[0].obb.conf.cpu().tolist())

        # -------------------------------------------------------------
        print(f'detected_conf: {outputs[0].obb.conf.cpu().tolist()} \
                detected_cls: {outputs[0].obb.cls.cpu().int().tolist()} \
                detected_xyxy: {outputs[0].obb.xyxyxyxy.cpu().int().tolist()} \
                detect_count:{len(outputs[0].obb.conf.cpu().tolist())}')

        # ------------------смотрим картинку-------------------------------------------
        # annotated_frame = outputs[0].plot()
        # annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        # plt.imshow(annotated_frame)
        # plt.show()
        # -------------------------------------------------------------
        return frame_element
