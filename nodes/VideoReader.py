from typing import Generator
import cv2
from elements.FrameElement import FrameElement
import numpy as np


class VideoReader:
    """Модуль для чтения кадров с видеопотока"""

    def __init__(self, config: dict) -> None:
        self.video_pth = config["src"]
        self.video_source = f"Processing of {self.video_pth}"
        self.stream = cv2.VideoCapture(self.video_pth)

    def process(self) -> Generator[FrameElement, None, None]:
        # номер кадра текущего видео
        frame_num = 0
        while True:
            ret, frame = self.stream.read()
            # print(frame.shape)
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]
            frame_num += 1
            source = self.video_pth
# ------------------------блок 'для просмотра видео'------
            # cv2.imshow('Webcam', frame)
            # # Выход из цикла по нажатию клавиши 'q'
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # cv2.waitKey(1)
# -----------------------конец блока 'для просмотра видео'------
            print(frame_num)
            yield FrameElement(source, frame, frame_num, frame_width, frame_height)
