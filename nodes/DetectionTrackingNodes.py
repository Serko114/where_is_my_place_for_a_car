from ultralytics import YOLO
import torch
import numpy as np

from utils_local.utils import profile_time
from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import VideoEndBreakElement
from byte_tracker.byte_tracker_model import BYTETracker as ByteTracker


class DetectionTrackingNodes:
    """Модуль инференса модели детекции + трекинг алгоритма"""

    def __init__(self, config) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Детекция будет производиться на {device}')

        config_yolo = config["detection_node"]
        self.model = YOLO(config_yolo["weight_pth"], task='detect')
        self.classes = self.model.names
        self.conf = config_yolo["confidence"]
        self.iou = config_yolo["iou"]
        self.imgsz = config_yolo["imgsz"]
        self.classes_to_detect = config_yolo["classes_to_detect"]
        self.dict_clsses = config_yolo["dict_clsses"]

        config_bytetrack = config["tracking_node"]

        # ByteTrack param
        first_track_thresh = config_bytetrack["first_track_thresh"]
        second_track_thresh = config_bytetrack["second_track_thresh"]
        match_thresh = config_bytetrack["match_thresh"]
        track_buffer = config_bytetrack["track_buffer"]
        fps = 30  # ставим равным 30 чтобы track_buffer мерился в кадрах
        self.tracker = ByteTracker(
            fps, first_track_thresh, second_track_thresh, match_thresh, track_buffer, 1
        )

    @profile_time
    def process(self, frame_element: FrameElement) -> FrameElement:
        # Выйти из обработки если это пришел VideoEndBreakElement а не FrameElement
        if isinstance(frame_element, VideoEndBreakElement):
            return frame_element
        assert isinstance(
            frame_element, FrameElement
        ), f"DetectionTrackingNodes | Неправильный формат входного элемента {type(frame_element)}"

        frame = frame_element.frame.copy()

        outputs = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False,
                                     iou=self.iou, classes=self.classes_to_detect)
        # print(f'OUTPUTS: {outputs}')
        frame_element.detected_conf = outputs[0].boxes.conf.cpu().tolist()
        detected_cls = outputs[0].boxes.cls.cpu().int().tolist()
        frame_element.detected_cls = [self.classes[i] for i in detected_cls]
        frame_element.detected_xyxy = outputs[0].boxes.xyxy.cpu(
        ).int().tolist()

        # Преподготовка данных на подачу в трекер
        detections_list = self._get_results_dor_tracker(outputs)
        # на выходе будет вот это: [[277.84, 68.415, 429.72, 289.89, 0.98837, 0]]
        # print(f'РЕЗУЛЬТИРУЮЩИЙ СПИСОК: {detections_list}')
        # Если детекций нет, то оправляем пустой массив
        if len(detections_list) == 0:
            detections_list = np.empty((0, 6))

        track_list = self.tracker.update(
            torch.tensor(detections_list), xyxy=True)
        # print(f'ID: {[int(t.track_id) for t in track_list]}')
        # Получение id list
        frame_element.id_list = [int(t.track_id) for t in track_list]

        # Получение box list
        frame_element.tracked_xyxy = [
            list(t.tlbr.astype(int)) for t in track_list]

        # Получение object class names
        frame_element.tracked_cls = [
            self.classes[int(t.class_name)] for t in track_list]

        # Получение conf scores
        frame_element.tracked_conf = [t.score for t in track_list]
        # -------------------------------------------------------------
        # создаем список с указанием единичкой на класс [0 1 0 0 0 0] (словарь см. в config.yaml)
        frame_element.cls_id = list(self.f(track_list))
        # print(self.f(track_list))
        # -------------------------------------------------------------
        print(f'id: {[int(t.track_id) for t in track_list]} \
                рамка: {[list(t.tlbr.astype(int)) for t in track_list]} \
                класс: {[self.classes[int(t.class_name)] for t in track_list]} \
                бинарный класс:{list(self.f(track_list))} \
                вероятность: {[t.score for t in track_list]}')
        # типа результат: [1] [[272, 63, 424, 289]] ['joyful'] [0 1 0 0 0 0] [0.9907557368278503]
        # -------------------------------------------------------------
        return frame_element

    def f(self, tr_lst) -> list:
        # функция для создания списка для подачи в столбцы эмоций в двоичном виде [0 1 0 0 0 0] (словарь см. в config.yaml)
        massive = np.array([int(i) for i in np.zeros(6)])
        classs_id = [self.classes[int(t.class_name)] for t in tr_lst][0]
        s = self.dict_clsses[classs_id]
        massive[s] = 1
        return massive

    def _get_results_dor_tracker(self, results) -> np.ndarray:
        # Приведение данных в правильную форму для трекера
        detections_list = []
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            # трекаем те же классы что и детектируем
            if class_id[0] in self.classes_to_detect:

                bbox = result.boxes.xyxy.cpu().numpy()
                confidence = result.boxes.conf.cpu().numpy()

                class_id_value = class_id[0]
                # class_id_value = (
                #     2  # Будем все трекуемые объекты считать классом car чтобы не было ошибок
                # )
                # print(confidence)
                # print(f'класс: {class_id_value} уверенность: {confidence[0]}')

                merged_detection = [
                    bbox[0][0],
                    bbox[0][1],
                    bbox[0][2],
                    bbox[0][3],
                    confidence[0],
                    class_id_value,
                ]
                # print(f'коробки: {merged_detection}')
                detections_list.append(merged_detection)
                # print(f'РЕЗУЛЬТАТ: {merged_detection}')
        # ----------------------------
        # отбирает одну детакцию с максимальным значением уверенности для одного id треккинга
        s = max([i[-2] for i in detections_list])
        detections_list = [i for i in detections_list if i[-2] == s]
        # ----------------------------
        # print(f'РЕЗУЛЬТАТ ОКОНЧАТЕЛЬНЫЙ: {detections_list}')
        return np.array(detections_list)
