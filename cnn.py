import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import os
import zipfile
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential



def main():

    full_path = os.path.abspath('datasets/animals') 
     # Corrected file path
    print(full_path)
    img_height = 128
    img_width = 128
    batch_size = 32

    # data_dir = tf.keras.utils.get_file('animals.zip', full_path, extract=True)
    train_ds = tf.keras.utils.image_dataset_from_directory(directory=full_path, labels='inferred', validation_split=0.15, subset="training", seed=123)


    test_ds = tf.keras.utils.image_dataset_from_directory(directory=full_path, labels='inferred', validation_split=0.15, subset="validation", seed=123, image_size=(img_height, img_width), batch_size=batch_size)

    print(train_ds.class_names)

if __name__ == "__main__":
    main()