import os
import cv2

# Changes all images to .jpg format and resizes them to 256x256. Images are also in RGB format
# Rev 2: added an index to both avoid file name collisions and to find some issues, does not affect resized images - JT
# Rev 3: added error handling for when images are not correct and reads in an incorrect image type, in this case, 
# skips it and displays the error, the current pizza types dataset contains said incorrect images - JT

def clean_data(data_dir):
    i = 0
    for dirpath, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if not filename.lower().endswith('.jpg'):
                old_image_path = os.path.join(dirpath, filename)
                image = cv2.imread(old_image_path)

                if image is None:
                    print(f"Error loading image for jpg conversion: {image_path}")
                    continue

                resized_image = cv2.resize(image, (256,256))

                new_image_path = os.path.splitext(old_image_path)[0] + f'__{i}.jpg'
                cv2.imwrite(new_image_path, resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                os.remove(old_image_path)
            else:
                image_path = os.path.join(dirpath, filename)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Error loading image for resizing: {image_path}")
                    continue

                resized_image = cv2.resize(image, (256,256))

                cv2.imwrite(image_path, resized_image)

            i = i + 1
