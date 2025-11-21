import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# Produces 10 augmented images for every 1 image in the dataset. The augmented images will be added to the dataset
def augment_data(data_dir):
    for pizza_type in os.listdir(data_dir):
        pizza_folder = os.path.join(data_dir, pizza_type)

        for filename in os.listdir(pizza_folder):
            image_path = os.path.join(pizza_folder, filename)

            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = np.expand_dims(img_rgb, axis=0)

            datagen = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

            augmented_images = []
            for j in range(10):
                augmented_image_rgb = datagen.flow(img_rgb, batch_size=1).next()[0]
                augmented_images.append(augmented_image_rgb)

                augmented_filename = f"augmented_{filename[:-4]}_{j}.jpg"
                augmented_file_path = os.path.join(pizza_folder, augmented_filename)
                cv2.imwrite(augmented_file_path, cv2.cvtColor(augmented_image_rgb, cv2.COLOR_BGR2RGB), [int(cv2.IMWRITE_JPEG_QUALITY), 100])