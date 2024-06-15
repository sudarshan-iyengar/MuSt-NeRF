import cv2
import os


def crop_and_downsample_image(image_path, left_crop, right_crop, top_crop, bottom_crop, scale_factor=2):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    # Get the height and width of the image
    height, width = image.shape[:2]
    
    # Crop the image
    cropped_image = image[top_crop:height-bottom_crop, left_crop:width-right_crop]
    height = height-bottom_crop*2
    width=width-right_crop*2
    print(width//scale_factor)
    print(height//scale_factor)
    
    # Downsample the cropped image
    downsampled_image = cv2.resize(cropped_image, (width // scale_factor, height // scale_factor))
    
    return downsampled_image
    
# Example usage:
#input_image_path = "1341.jpg"
image_path_list = ["1781.jpg"]
left_crop = 24
right_crop = 24
top_crop = 16
bottom_crop = 16

output_folder = "im_crop"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
for input_image_path in image_path_list:
	output_image = crop_and_downsample_image(input_image_path, left_crop, right_crop, top_crop, bottom_crop)
	output_file_name = os.path.join(output_folder, input_image_path.split('.')[0] + ".jpg")  # Specify the folder path
	#output_file_name = input_image_path.split('.')[0] + "_cr.jpg"
	cv2.imwrite(output_file_name, output_image)

