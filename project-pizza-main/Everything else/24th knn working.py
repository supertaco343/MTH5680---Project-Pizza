import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

DATA_DIR = 'pizza types'
CLASSES = ['cheese', 'pepperoni', 'sausage', 'hawaiian', 'margherita']
CLASS_MAP = {CLASSES[i] : i for i in range(len(CLASSES)) }

image_paths = []

for dirpath, _, filenames in os.walk(DATA_DIR):
    for filename in filenames:
        image_path = os.path.join(dirpath, filename)

        # Try to load the image
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error loading image for jpg conversion: {image_path}")
            continue

        # Check if the image is already in JPG format
        if not image_path.lower().endswith('.jpg'):
            # Create a new path with the .jpg extension
            new_image_path = os.path.splitext(image_path)[0] + '.jpg'

            # Try to write the image in JPG format
            success = cv2.imwrite(new_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            if not success:
                print(f"Error converting to JPG: {image_path}")
                continue

            # Remove the old image
            os.remove(image_path)

            # Update the image path to the new JPG path
            image_paths.append(new_image_path)
        else:
            # The image is already in JPG format
            image_paths.append(image_path)

print("Image Paths:", image_paths)
            
# Resizing each image
for image_path in image_paths:
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error loading image for resizing: {image_path}")
        continue
    
    resized_image = cv2.resize(image, (256, 256))
    cv2.imwrite(image_path, resized_image)
    
print(image_paths[2])
img = cv2.imread(image_paths[2])

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

for pizza_type in os.listdir(DATA_DIR):
    pizza_folder = os.path.join(DATA_DIR, pizza_type)

    for filename in os.listdir(pizza_folder):
        image_path = os.path.join(pizza_folder, filename)

        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Error loading image for augmentation: {image_path}")
            continue
        
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


X = []
y = []

for pizza_type in os.listdir(DATA_DIR):
    pizza_folder_path = os.path.join(DATA_DIR, pizza_type)
    for filename in os.listdir(pizza_folder_path):
        image_path = os.path.join(pizza_folder_path, filename)
        
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Error loading image for data1: {image_path}")
            continue
        
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if image is None:
            print(f"Error loading image for data2: {image_path}")
            continue
        
        X.append(image)
        y.append(pizza_type)

X = np.array(
    list(
        map(lambda img: img * 1./255, X)
    )
)
y = np.array(
    list(
        map(lambda pizza_type: CLASS_MAP[pizza_type], y)
    )
)

X = np.reshape(X, (X.shape[0], -1))

for k in range(1, 101):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)

    y_pred = model.predict(X)
    print('For k =', k)
    print(accuracy_score(y, y_pred))
    print()
    
