import numpy as np
import time


class FrameElement:
    # Класс, содержаций информацию о конкретном кадре видеопотока
    def __init__(
        self,
        source: str,
        frame: np.ndarray,
        frame_num: float,
        frame_width: int,
        frame_height: int,
        frame_result: np.ndarray | None = None,
        # detection
        detected_conf: list | None = None,
        detected_cls: list | None = None,
        detected_xyxy: list[list] | None = None,
        tracked_conf: list | None = None,
        tracked_cls: list | None = None,
        tracked_xyxy: list[list] | None = None,
        id_list: list | None = None,
        cls_id: list | None = None,
        # далее


    ) -> None:
        self.source = source  # Путь к видео или номер камеры с которой берем поток
        self.frame = frame  # Кадр bgr формата
        self.frame_num = frame_num  # Нормер кадра с потока
        self.frame_width = frame_width  # ширина кадра
        self.frame_height = frame_height  # ширина кадра
        self.frame_result = frame_result  # Итоговый обработанный кадр
        # ------detection-------
        # Список уверенностей задетектированных объектов
        self.detected_conf = detected_conf
        self.detected_cls = detected_cls  # Список классов задетектированных объектов
        self.detected_xyxy = detected_xyxy  # Список списков с координатами xyxy боксов
        # Результаты корректировки трекинг алгоритмом:
        self.tracked_conf = tracked_conf  # Список уверенностей задетектированных объектов
        self.tracked_cls = tracked_cls  # Список классов задетектированных объектов
        self.tracked_xyxy = tracked_xyxy  # Список списков с координатами xyxy боксов
        self.id_list = id_list  # Список обнаруженных id трекуемых объектов
        # Список по числу классов, двоичная хрень, типа [0, 0, 1, 0, 0, 0] для подачи в DB
        self.cls_id = cls_id
        # -----------------------
        self.send_info_of_frame_to_db = True
