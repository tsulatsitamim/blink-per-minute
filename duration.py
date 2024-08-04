import cv2

def get_video_duration_opencv(file_path):
    video = cv2.VideoCapture(file_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    return duration

duration = get_video_duration_opencv("./0802.mp4")
print(f"Duration: {duration} seconds")