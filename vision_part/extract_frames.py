import cv2
import os


class VideoFrameExtractor:
    def __init__(self, 
                 output_dir: str = "cache_frames", 
                 video_source: str = "0", 
                 frame_interval: int = 10):
        """
        Initialize the video frame extractor.

        Args:
            output_dir (str): Directory to save extracted frames.
            video_source (str): Path to video file or livestream source (e.g., '0' for webcam).
            frame_interval (int): Number of frames to skip between extractions.
        """
        self.output_dir = output_dir
        self.video_source = int(video_source) if video_source.isdigit() else video_source
        self.frame_interval = frame_interval

        os.makedirs(self.output_dir, exist_ok=True)
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {video_source}")

        mode = "Live Stream" if isinstance(self.video_source, int) else "Video File"
        print(f"[Initialized] Mode: {mode}, Source: {video_source}")
        print(f"[Config] Saving every {self.frame_interval} frame(s) to: {self.output_dir}")

    def extract_frames(self, max_frames: int = None):
        """
        Extract frames from video and save every `frame_interval`-th frame.

        Args:
            max_frames (int, optional): Max frames to save. None = until end.
        """
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[Done] End of stream or failure.")
                break

            if frame_count % self.frame_interval == 0:
                filename = os.path.join(self.output_dir, f"frame_{frame_count:05d}.jpg")
                cv2.imwrite(filename, frame)
                print(f"[Saved] {os.path.basename(filename)}")
                saved_count += 1
                if max_frames and saved_count >= max_frames:
                    break

            #else:
                #print(f"[Skipped] Frame {frame_count} (interval: {self.frame_interval})")

            frame_count += 1

        self.cap.release()
        print(f"[Finished] Total saved frames: {saved_count}, Total processed frames: {frame_count}")


if __name__ == "__main__":
    aria_video_path = "test_video/cup1.mp4"
    ur5e_video_path = "test_video/cup_good.mp4"

    aria_output_dir = "aria_images"
    ur5e_output_dir = "ur5e_images_scene/images"

    print("\n[Start] Extracting frames from Aria video...")
    aria_extractor = VideoFrameExtractor(
        video_source=aria_video_path,
        output_dir=aria_output_dir,
        frame_interval=30
    )
    aria_extractor.extract_frames()

    print("\n[Start] Extracting frames from UR5e video...")
    ur5e_extractor = VideoFrameExtractor(
        video_source=ur5e_video_path,
        output_dir=ur5e_output_dir,
        frame_interval=30
    )
    ur5e_extractor.extract_frames()
