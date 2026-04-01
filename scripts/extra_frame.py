import cv2
import os

def extract_frames_per_second(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)  # 1 frame per second

    print(f"Video FPS: {fps}")
    print(f"Saving 1 frame every {frame_interval} frames")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            output_path = os.path.join(
                output_folder,
                f"frame_{saved_count:05d}.jpg"
            )
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done! Saved {saved_count} frames.")


video_path = input("Enter video path: ").strip()
output_folder = input("Enter output folder for frames: ").strip()

extract_frames_per_second(video_path, output_folder)