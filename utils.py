import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def show(img):
    plt.figure(figsize=(100,100))
    plt.imshow(img, cmap="gray")
    plt.show()

def ttoi(tensor):
    img = tensor.cpu().numpy()
    return img

def visualization(images, num_images=20, num_rows=2):
    r = num_rows
    i = num_images

    fig = plt.figure(figsize=(30,10))
    for index in np.arange(20):
        ax = fig.add_subplot(r, i/r, index+1, xticks=[], yticks=[])
        ax.imshow(images[index], cmap="gray")

def save_samples_images(images, image_path, num_images=20, num_rows=2, height=28, width=28):
    concat_images = concatenate_images(images, num_images, num_rows, height, width)
    cv2.imwrite(image_path, concat_images)

def concatenate_images(images, num_images=20, num_rows=2, height=28, width=28):
    # Compute Necessary Dimensions
    pixel_row = num_rows * height
    pixel_col = (num_images // num_rows) * width
    num_cols = num_images // num_rows

    # Placeholder Image
    placeholder = np.empty([pixel_row, pixel_col])
    
    # Reshape
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            placeholder[i*height:(i+1)*height, j*width:(j+1)*width] = images[index]

    return placeholder * 255
