
import os
import cv2

def clean_data(data_dir):
    for dirpath, _, filenames in os.walk(data_dir):
        for filename in filenames:
            image_path = os.path.join(dirpath, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error loading image: {image_path}")
                continue

            if not filename.lower().endswith('.jpg'):
                new_image_path = os.path.splitext(image_path)[0] + '_converted.jpg'
                cv2.imwrite(new_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                os.remove(image_path)

data_dir = 'pizza types'
clean_data(data_dir)
