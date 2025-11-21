import os
import shutil

def full_copy(src, dest):
    try:
        # Copy the entire directory and its contents to the destination
        shutil.copytree(src, dest)
        print(f"Successfully copied from {src} to {dest}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage:
#source_directory = "/path/to/source"
#destination_directory = "/path/to/destination"

#copy_directory(source_directory, destination_directory)

def partial_copy(src, dest, num_files):
    try:
        # Iterate through each subdirectory in the source directory
        for root, dirs, files in os.walk(src):
            for subdir2 in dirs:
                for subdir in subdir2:
                    src_subdir = os.path.join(root, subdir)
                    dest_subdir = os.path.join(dest, os.path.relpath(src_subdir, src))

                    # Create the corresponding subdirectory in the destination if it doesn't exist
                    try:
                        os.makedirs(dest_subdir, exist_ok=True)
                    except Exception as e:
                        print(f"Error creating destination directory {dest_subdir}: {e}")
                        continue

                    # List all files in the source subdirectory
                    file_list = os.listdir(src_subdir)

                    # Shuffle the file list to randomize the selection
                    #random.shuffle(file_list)
                    #unneeded for now, but here in case we want to do it
                
                    # Copy a specified number of files from the source to the destination
                    for file_name in file_list[:num_files]:
                        src_file = os.path.join(src_subdir, file_name)
                        dest_file = os.path.join(dest_subdir, file_name)

                        try:
                            shutil.copy2(src_file, dest_file)
                        except Exception as e:
                            print(f"Error copying {src_file} to {dest_file}: {e}")

            print(f"Successfully copied {num_files} files from each subdirectory from {src} to {dest}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage:
#source_directory = "/path/to/source"
#destination_directory = "/path/to/destination"
#number_of_files_to_copy = 10

#copy_files(source_directory, destination_directory, number_of_files_to_copy)


#pulled from GPT 

######NOTE: for some reason, the partial_copy does not like 