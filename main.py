import hydra
from nodes.VideoReader import VideoReader
from nodes.DetectionTrackingNodes import DetectionTrackingNodes
from nodes.VideoShow import VideoShowDetection
from nodes.VideoSaverNode import VideoSaverNode
from nodes.SendInfoDBNode import SendInfoDBNode
# import cv2


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config) -> None:
    video_reader = VideoReader(config["video_reader"])
    detection_node = DetectionTrackingNodes(config)
    show_detection_node = VideoShowDetection(config)
    # SendInfoDBNode(config)

    save_video = config["pipeline"]["save_video"]
    send_info_db = config["pipeline"]["send_info_db"]
    video_show = config["pipeline"]["video_show"]
    if send_info_db:
        send_info_db_node = SendInfoDBNode(config)
    if video_show:
        show_detection_node = VideoShowDetection(config)
    if save_video:
        video_saver_node = VideoSaverNode(config["video_saver_node"])

    for frame_element in video_reader.process():
        frame_element = detection_node.process(frame_element)
        # print('FUCK111!')

        if send_info_db:
            frame_element = send_info_db_node.process(frame_element)
        # print('FUCK222!')
        # frame_element = show_detection_node.process(frame_element)
        if video_show:
            show_detection_node.process(frame_element)
        if save_video:
            video_saver_node.process(frame_element)
        # print('iiiiiiiiii')
        # print()


if __name__ == "__main__":
    main()
