#!/usr/bin/python3

# script for cropping images, creating blocks and saving them to a new folder
import cv2
import numpy as np
import os
import sys
from pathlib import Path

# crops the image into a square of given dimensions
def cropImage(img, dim=(128,128)):
    width, height = img.shape[1], img.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    cropped_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return cv2.resize(cropped_img, (int(img.shape[1]), int(img.shape[0])))

# creates and returns an n x n grid of images from an image
def splitImage(img, n):
    # get the image dimensions
    height, width, _ = img.shape
    img_height = height // n
    img_width = width // n
    images = []
    for i in range(n):
        for j in range(n):
            # store subset of the image in the list as a separate item
            images.append(img[i*img_height:(i+1)*img_height, j*img_width:(j+1)*img_width])
    return images


def main():

    # get path to the animals directory from first arg
    # get absolute path to the directory
    cw = os.getcwd()
    parent = Path(cw).parent
    path = str(parent) + '/datasets/animals'

    blocks = bool(sys.argv[1])
    num_blocks = int(sys.argv[2])

    # uncomment and change path below in case parsing arguments doesn't work
    # path = '/Users/gustavozunigapadron/Desktop/animals'
    # num_blocks = 4

    for _, dirs, _ in os.walk(path):
        for animal in dirs:
           for _, _, files in os.walk(os.path.join(path, animal)):
               for file in files:
                   
                   # read and crop image
                   image = cv2.imread(os.path.join(path, animal, file))
                   cropped_img = cropImage(image)
                   if not blocks:
                       cv2.imwrite(f'{path}/{animal}/{file[:-4]}_cropped.png', cropped_img)
                   else:
                       os.mkdir(f'{path}/{animal}/{file[:-4]}')
                       # split image into blocks
                       img_blocks = splitImage(cropped_img, num_blocks)

                        # save splits to the new folder
                       for i, img_block in enumerate(img_blocks):
                          cv2.imwrite(f'{path}/{animal}/{file[:-4]}/block{i}.png', img_block)

if __name__ == "__main__":
    main()