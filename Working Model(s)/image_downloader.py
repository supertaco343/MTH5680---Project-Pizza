from bing_image_downloader import downloader
import os

# Directory where data will be stored
DATA_DIR = 'pizza types'

if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)

# Download images for each class from the web
downloader.download('Cheese pizza',      limit=100, output_dir=DATA_DIR, timeout=300, verbose=False)
downloader.download('Pepperoni pizza',   limit=100, output_dir=DATA_DIR, timeout=300, verbose=False)
downloader.download('Hawaiian pizza',    limit=100, output_dir=DATA_DIR, timeout=300, verbose=False)
downloader.download('Black Olive pizza', limit=100, output_dir=DATA_DIR, timeout=300, verbose=False)
downloader.download('Taco pizza',        limit=100, output_dir=DATA_DIR, timeout=300, verbose=False)

# Change sub directory names
SUB_DIRS = os.listdir(DATA_DIR)
for sub_dir in SUB_DIRS:
    old_path = os.path.join(DATA_DIR, sub_dir)
    new_path = os.path.join(DATA_DIR, sub_dir.removesuffix(' pizza'))
    os.replace(old_path, new_path)
