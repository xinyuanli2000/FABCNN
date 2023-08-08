from PIL import Image
import os
import random

def change_color(image, new_color):
    # Get the center of the image
    width, height = image.size
    center_x, center_y = width // 2, height // 2

    # Define the radius for the region of interest (ROI)
    radius = random.randint(20, 50)

    # Check if the radius is too large for the image
    if center_x - radius < 0 or center_x + radius >= width or center_y - radius < 0 or center_y + radius >= height:
        print("Radius is too large for the image. Skipping modification.")
        return image

    # Get the pixels of the image
    pixels = image.load()

    # Loop through the ROI and change the color
    for y in range(center_y - radius, center_y + radius):
        for x in range(center_x - radius, center_x + radius):
            # Check if the pixel is within the circular ROI
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                # Set the new color for each pixel in the ROI
                pixels[x, y] = new_color

    return image

def process_images_in_folder(input_folder, output_folder):
    # List all files in the folder and its subfolders
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.JPEG'):  # Check if the file is a PNG image
                input_image_path = os.path.join(root, file)
                output_image_path = os.path.join(output_folder, os.path.relpath(input_image_path, input_folder))

                # Create output folder if it doesn't exist
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

                # Generate a random color for each image
                new_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                # Open the image using PIL
                image = Image.open(input_image_path)

                # Process the image and save it
                modified_image = change_color(image, new_color)
                modified_image.save(output_image_path)

                print(f"Modified image saved at: {output_image_path}")

# Example usage
if __name__ == "__main__":
    input_folder_path = "D:/study/msc/project/feedback-attention-cnn-main/dataset/Test_name"
    output_folder_path = "D:/study/msc/project/feedback-attention-cnn-main/dataset/test_color"

    process_images_in_folder(input_folder_path, output_folder_path)
