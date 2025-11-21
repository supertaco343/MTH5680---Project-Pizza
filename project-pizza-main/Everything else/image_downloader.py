from bing_image_downloader import downloader
import os

# Download images for each class from the web
downloader.download('pepperoni pizza', limit=20, output_dir='pizza types', verbose=False)
downloader.download('sausage pizza', limit=20, output_dir='pizza types', verbose=False)
downloader.download('hawaiian pizza', limit=20, output_dir='pizza types', verbose=False)
downloader.download('margherita pizza', limit=20, output_dir='pizza types', verbose=False)
downloader.download('cheese pizza', limit=20, output_dir='pizza types', verbose=False)

# Change sub directory names
main_directory = 'pizza types'
sub_directories = os.listdir(main_directory)

for sub_dir in sub_directories:
    old_path = os.path.join(main_directory, sub_dir)
    new_path = os.path.join(main_directory, sub_dir.removesuffix(' pizza'))
    os.replace(old_path, new_path)
