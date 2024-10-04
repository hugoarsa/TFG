import os
from PIL import Image

def resize_images_in_directory(directory, target_size=(512, 512)):
    # Loop through all files in the specified directory
    for filename in os.listdir(directory):
        # Get the full path of the file
        file_path = os.path.join(directory, filename)
        
        # Check if it's a file and has a valid image extension
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                # Open the image
                with Image.open(file_path) as img:
                    # Resize the image
                    img_resized = img.resize(target_size, Image.LANCZOS)
                    
                    # Save the resized image, overwriting the original
                    img_resized.save(file_path)
                    
                print(f"Resized and saved: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Specify the directory containing the images
directory_path = './images'
resize_images_in_directory(directory_path)