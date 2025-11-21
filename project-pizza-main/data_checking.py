from pathlib import Path
import imghdr

# checks if the data (in this case, images) will be supported by tensorflow, as some data
# is malformed/corrupt or formatted in an incorrect way.  
# most of this should be fixed during the data cleanup function, but this is here in case

# pulled from this post and slightly edited: 
# https://stackoverflow.com/questions/68191448/unknown-image-file-format-one-of-jpeg-png-gif-bmp-required

def check_data_type(data_dir):
    i = 0

    # data_dir = "Manually Sorted Pizza"
    image_extensions = [".png", ".jpg"]  # add there all your images file extensions

    img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
    for filepath in Path(data_dir).rglob("*"):
        if filepath.suffix.lower() in image_extensions:
            img_type = imghdr.what(filepath)
            if img_type is None:
                print(f"{filepath} is not an image")
                i = i + 1
            elif img_type not in img_type_accepted_by_tf:
                print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
                i = i + 1
                
    if i != 0:
        print("Number of unsupported images: ", i)
