import os
import cv2

# Path to the folder containing RGB images
folder_path = r"D:\M Tech\PDWS\SLRS\splitdataset128x128\splitdataset128x128\val\Z"



# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Iterate through each file
for file_name in file_list:
    # Construct the full path of the file
    file_path = os.path.join(folder_path, file_name)
    
    # Read the image
    image = cv2.imread(file_path)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Write the grayscale image back to the same file
    cv2.imwrite(file_path, gray_image)

print("Conversion complete.")
