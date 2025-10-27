import cv2
import os

def extract_frames(video_path, output_dir, every_n_frames=30):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n_frames == 0:
            frame_path = os.path.join(output_dir, f"{saved}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        count += 1
    cap.release()
    print(f"[✔] {os.path.basename(video_path)}: Saved {saved} frames.")

def process_all_repos(root_dir, output_root, every_n_frames=30):
    for i in range(1000):
        repo_name = f"{i:04d}"
        video_path = os.path.join(root_dir, repo_name, f"{repo_name}.mp4")
        output_dir = os.path.join(output_root, f"{i}")

        if os.path.exists(video_path):
            extract_frames(video_path, output_dir, every_n_frames)
        else:
            print(f"[!] Skipping {repo_name}: Video not found.")

# Example usage
if __name__ == "__main__":
    ROOT_REPOS_DIR = "videos/"       
    OUTPUT_DATASET_DIR = "multicolor_dataset/train" 
    FRAME_INTERVAL = 10                      

    process_all_repos(ROOT_REPOS_DIR, OUTPUT_DATASET_DIR, FRAME_INTERVAL)
