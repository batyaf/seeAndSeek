import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from skimage import io
from pathlib import Path
import os


def imageGenerator(path, num):
    # defining the changes for condensation
    imageGen = ImageDataGenerator(
        zoom_range=[0.9, 1.1],
        fill_mode='reflect',
        shear_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    imagePath = path
    myFile = Path(imagePath)

    if myFile.is_file():
        print('ok')
    else:
        print('no image')
        return None  # Return None if the image file does not exist

    # Read the image
    image = io.imread(imagePath)

    # Ensure the image has 3 channels (for RGB) even if it's grayscale
    image = np.expand_dims(image, axis=-1) if len(image.shape) == 2 else image

    # Expand dimensions to add batch size
    image = np.expand_dims(image, axis=0)

    # Generate augmented images
    augIter = imageGen.flow(image, batch_size=num)

    # Retrieve augmented images
    augImages = [next(augIter)[0].astype(np.uint8) for i in range(num)]
    return augImages


def saveImage(images, path, char_img):
    # Create the 'generator_img' folder if it doesn't exist
    generator_folder = os.path.join(path, 'generator_img')
    os.makedirs(generator_folder, exist_ok=True)

    char_img_name = os.path.splitext(char_img)[0]  # Remove the ".png" extension

    for i, img in enumerate(images, start=1):
        data = Image.fromarray(img.squeeze())
        nameImage = os.path.join(generator_folder, f"{char_img_name}_{i}.png")
        data.save(nameImage)

if __name__ == '__main__':
    import cv2
    # from reSize.reSizeImage import resizeWithWhiteBackground
    data_folders = [data_letter for data_letter in os.listdir("./data")]
    for data_letter_folder in data_folders:
        img_files = [file for file in os.listdir(f"./data/{data_letter_folder}")]
        for img_file in img_files:
            # Extract font name from the file
            print(img_file)
            char_img_files = [file for file in os.listdir(f"./data/{data_letter_folder}/{img_file}")]
            for char_img in char_img_files:
                print(char_img)
                s = imageGenerator(f"./data/{data_letter_folder}/{img_file}/{char_img}", 40)
                if s is not None:
                    saveImage(s, f"./data/{data_letter_folder}/{img_file}", char_img)
                    print(f"Augmented images saved for {char_img}")
                else:
                    print(f"Skipping {char_img} due to image loading or augmentation failure")



