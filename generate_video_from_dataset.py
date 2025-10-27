import cv2
import numpy as np
import magical, glob
from absl import flags
from absl import app
from torchkit import experiment
import os
import cv2
from PIL import Image

FLAGS = flags.FLAGS

flags.DEFINE_list("seeds", [0,25], "List specifying the range of seeds to run.")

def create_video_from_images(image_array, output_path, fps=30):
    """
    Create a video from an array of images.
    
    :param image_array: List of images (each image as a numpy array).
    :param output_path: Path where the output video will be saved.
    :param fps: Frames per second (default: 30).
    """
    # Get the height, width of the first image
    height, width, layers = image_array[0].shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For saving as MP4
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write each frame to the video
    for img in image_array:
        video.write(img)
    
    # Release the video writer object
    video.release()
    print(f"Video saved at {output_path}")

@experiment.pdb_fallback
def main(_):
    # Load from Magical dataset
    save_path = "/tmp/xirl/datasets/magical_xirl/cluster_colour/video"
    save_path_images = "/tmp/xirl/datasets/magical_xirl/cluster_colour/train/gripper"
    height, width = (384, 384)
    for int_seed in range(*list(map(int, FLAGS.seeds))):
        seed = str(int_seed).zfill(3)
        video_path = f"/home/lucaianniello/Thesis/MagicalDataset/magical/demos/cluster-colour/demo-ClusterColour-Demo-v0-{seed}.pkl.gz"
        print(f"Creating video from {video_path}")
        traj = list(magical.load_demos(glob.glob(video_path)))
        '''
        traj[0]:
            - 'trajectory'
                [0]: actions
                [1]: images
                    [i]: image
                        - 'allo'
                        - 'ego'
                [2]: rewards
                [3]: eval_score
            - 'score'
            - 'env_name'
        '''
        img_list = []
        save_path_images_seed = f"{save_path_images}/{int_seed}"
        os.makedirs(save_path_images_seed, exist_ok=True)
        for i in range(len(traj[0]['trajectory'][1])):
            img = traj[0]['trajectory'][1][i]['allo']
            img_list.append(img)

            if img.shape[:2] != (height, width):
                img = cv2.resize(
                    img,
                    dsize=(width, height),
                    interpolation=cv2.INTER_CUBIC,
                )
            Image.fromarray(img).save(f"{save_path_images_seed}/{i}.png")


        # Create video from images
        create_video_from_images(img_list, f"{save_path}/output_video_{seed}.mp4", fps=30)


# Example usage
if __name__ == "__main__":
    app.run(main)