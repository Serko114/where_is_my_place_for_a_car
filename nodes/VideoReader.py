from typing import Generator
import cv2
from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import FrameElement
import numpy as np
import datetime


class VideoReader:
    """Модуль для чтения кадров с видеопотока"""

    def __init__(self, config: dict) -> None:
        self.video_pth = config["src"]  # путь до видео
        # self.video_source = f"Processing of {self.video_pth}"
        self.stream = cv2.VideoCapture(self.video_pth)  # чтение видео
        self.date_time = str(datetime.datetime.now()).split(
            ' ')  # ['2025-11-09', '20:22:20.889841'] вытаскивает текущее время

    def process(self) -> Generator[FrameElement, None, None]:
        # номер кадра текущего видео
        frame_num = 0
        while True:
            ret, frame = self.stream.read()  # покадровое чтение видео
            # print(frame.shape)
            # print(frame.dtype)
            # print(frame)
            # меняет каналы местами
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_width = frame.shape[1]  # ширина картинки
            frame_height = frame.shape[0]  # высота картинки
            frame_num += 1  # нумерация кадров
            source = self.video_pth  # путь до видео
            # вытаскивает дату из списка (см. выше) -> '2025-11-09' str
            date = self.date_time[0]
            # print(self.date_time)
            # print(f'-------------------------------------------{date}')
            time = self.date_time[-1].split('.')[0]
            # вытаскивает время из списка (см. выше) -> '20:22:20' str, удаляя миллисекунды
            # print(f'-------------------------------------------{time}')
# ------------------------блок 'для просмотра видео'------
            # cv2.imshow('Webcam', frame)
            # # Выход из цикла по нажатию клавиши 'q'
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # cv2.waitKey(1)
# -----------------------конец блока 'для просмотра видео'------
            # print(f'frame: {frame.dtype}')
            yield FrameElement(source, frame, frame_num, frame_width, frame_height, date, time)
            # ресурс, кадр (массив array (1080, 1920, 3) uint8), номер_кадра, ширина, высота, дата, время
# В таком виде подается картинка:
#  [[ 51 134 110]
#   [ 51 134 110]
#   [ 51 134 110]
#   ...
#   [ 74  55  54]
#   [ 71  52  51]
#   [ 71  52  51]]

#  [[ 50 133 109]
#   [ 50 133 109]
#   [ 50 133 109]
#   ...
#   [ 73  54  53]
#   [ 71  52  51]
#   [ 71  52  51]]

#  [[ 50 133 109]
#   [ 50 133 109]
#   [ 50 133 109]
#   ...
#   [ 73  54  53]
#   [ 71  52  51]
#   [ 71  52  51]]]
