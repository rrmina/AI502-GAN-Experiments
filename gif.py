import os
from imageio import imread, mimsave

folder = "results/"
images = []
IMAGE_BASE_FILE_NAME = "epoch"
IMAGE_BASE_FILE_TYPE = ".png"

def make_gif(frames_path, save_name, fps):
    # Extract image paths. Natural sorting of directory list
    # Unfortunately, Python does not have a native support for natural sorting :()
    base_name_len = len(IMAGE_BASE_FILE_NAME)
    filetype_len = len(IMAGE_BASE_FILE_TYPE)
    filenames = [img for img in sorted(os.listdir(frames_path), key=lambda x : int(x[base_name_len:-filetype_len])) if img.endswith(IMAGE_BASE_FILE_TYPE)]
    
    # Read and store image files
    for file_name in filenames:
        images.append(imread(frames_path + file_name))

    # Save as one gif file
    mimsave(save_name + ".gif", images, fps=fps)
    
make_gif(folder, folder[:-1], 10)