import os
import subprocess
from multiprocessing import Pool, cpu_count

download_dir = "videos"
standard_dir = "standardized_videos"

TARGET_WIDTH = 640
TARGET_HEIGHT = 360
TARGET_FPS = 30

os.makedirs(standard_dir, exist_ok=True)

def standardize_video_ffmpeg(paths):
    input_path, output_path = paths
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    command = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"scale=w={TARGET_WIDTH}:h={TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
               f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,"
               f"fps={TARGET_FPS}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "aac",
        output_path
    ]

    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"[✓] {input_path} → {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[!] Error processing {input_path}: {e}")

def gather_video_paths():
    tasks = []
    for style_folder in os.listdir(download_dir):
        input_style_path = os.path.join(download_dir, style_folder)
        output_style_path = os.path.join(standard_dir, style_folder)

        if not os.path.isdir(input_style_path):
            continue

        for file in os.listdir(input_style_path):
            if file.endswith(".mp4"):
                input_file = os.path.join(input_style_path, file)
                output_file = os.path.join(output_style_path, file)
                tasks.append((input_file, output_file))
    return tasks

def main():
    all_tasks = gather_video_paths()
    print(f"[i] Found {len(all_tasks)} videos to standardize using {cpu_count()} cores...")

    with Pool(cpu_count()) as pool:
        pool.map(standardize_video_ffmpeg, all_tasks)

    print("[✓] All videos standardized.")

if __name__ == "__main__":
    main()
