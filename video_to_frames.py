import os
import cv2

def video_to_frames(video_path, output_dir, frame_interval=1, resize=None):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    print(f"Saving every {frame_interval} frame(s)")

    saved_count = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            if resize:
                frame = cv2.resize(frame, resize)
            frame_path = os.path.join(output_dir, f"{saved_count}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_index += 1

    cap.release()
    print(f"Done! Saved {saved_count} frames to {output_dir}")

if __name__ == "__main__":
    # ========= 你只需要修改这几行 =========
    video_path   = "/home/sz/smartgrip/ball.mp4"   # 输入视频路径
    output_dir   = "/home/sz/smartgrip/outputs/ball"                        # 输出图像帧目录
    frame_interval = 5                               # 每隔多少帧保存一帧
    resize = None #(640, 480)                              # 是否缩放图像尺寸，如不缩放设为 None
    # =======================================

    video_to_frames(video_path, output_dir, frame_interval, resize)
