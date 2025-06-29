from typing import Generator
import cv2
import random
from elements.FrameElement import FrameElement
from utils_local.utils import profile_time, FPS_Counter
# from FrameElement import FrameElement
import numpy as np


class VideoShowDetection:
    """Модуль для чтения кадров с видеопотока"""

    def __init__(self, config: dict) -> None:
        config_show_node = config["show_node"]
        self.scale = config_show_node["scale"]
        self.imshow = config_show_node["imshow"]
        self.show_only_yolo_detections = config_show_node["show_only_yolo_detections"]
        self.show_track_id_different_colors = config_show_node["show_track_id_different_colors"]
        self.show_info_statistics = config_show_node["show_info_statistics"]
        self.russians_classes = config_show_node["show_russians_classes"]
        self.fps_counter_N_frames_stat = config_show_node['fps_counter_N_frames_stat']
        # self.file_save_video = config_show_node['folder_save_video']
        # Параметры для шрифтов:
        self.fontFace = fontFace = cv2.FONT_HERSHEY_COMPLEX  # 1
        self.fontScale = 0.5
        self.thickness = 1
        # Параметры для полигонов и bboxes:
        self.thickness_lines = 2
        # Параметры для экрана статистики:
        self.width_window = 700  # ширина экрана в пикселях
        # self.save_video = config_show_node["save_video"]

    def process(self, frame_element: FrameElement, fps_counter=None) -> FrameElement:
        # --------блок - смоткрим видео без рамки------------------
        # frame = frame_element.frame.copy()
        # frame_number = frame_element.frame_num
        # print(frame_number)
        # cv2.imshow(f'Webcam', frame)
        # cv2.waitKey(1)
        # -----------------------------------------------------------------
        frame_result = frame_element.frame.copy()
        # Отображение лишь результатов детекции:
        if self.show_only_yolo_detections:
            for box, class_name in zip(frame_element.detected_xyxy, frame_element.detected_cls):
                x1, y1, x2, y2 = box
                # Отрисовка прямоугольника
                cv2.rectangle(frame_result, (x1, y1), (x2, y2), (0, 0, 0), 2)
                # Добавление подписи с именем класса
                cv2.putText(
                    frame_result,
                    class_name,
                    (x1, y1 - 10),
                    fontFace=self.fontFace,
                    fontScale=self.fontScale,
                    thickness=self.thickness,
                    color=(0, 0, 255),
                )
        else:
            # Отображение результатов трекинга:
            for box, class_name, id in zip(
                frame_element.tracked_xyxy, frame_element.tracked_cls, frame_element.id_list
            ):
                x1, y1, x2, y2 = box
                # Отрисовка прямоугольника

                random.seed(int(id))
                color = (random.randint(0, 255), random.randint(
                    0, 255), random.randint(0, 255))

                cv2.rectangle(frame_result, (x1, y1), (x2, y2),
                              color, self.thickness_lines)
                # Добавление подписи с именем класса
                cv2.putText(
                    frame_result,
                    f"{self.russians_classes[frame_element.tracked_cls[0]]}",
                    (x1, y1 - 10),
                    fontFace=self.fontFace,
                    fontScale=self.fontScale,
                    thickness=self.thickness,
                    color=(0, 0, 255),
                )
        frame_element.frame_result = frame_result
        frame_show = frame_result
        # frame_show = cv2.resize(frame_result.copy(),
        #                         (-1, -1), fx=self.scale, fy=self.scale)
        # --------------- блок записи видео для readme ------------------
        # if self.save_video:
        #     # fps, ширина, высота
        #     fps = int(self.fps_counter_N_frames_stat)
        #     frame_width = int(frame_result.get(cv2.CAP_PROP_FRAME_WIDTH))
        #     frame_height = int(frame_result.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #     # определение параметров записи
        #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #     out = cv2.VideoWriter(self.file_save_video, fourcc,
        #                           fps, (frame_width, frame_height))
        # if self.save_video:
        #     out.write(frame_show)
        # ---------------------------------------------------------------
        if self.imshow:
            cv2.imshow(frame_element.source, frame_show)
            cv2.waitKey(1)

        return frame_element
