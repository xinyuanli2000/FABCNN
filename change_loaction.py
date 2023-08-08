from PIL import Image
import numpy as np
import os
import random

def find_background_color(image_path):
    image = Image.open(image_path)

    image_array = np.array(image)

    background_color = tuple(np.mean(image_array.reshape(-1, 3), axis=0).astype(int))

    return background_color

def translate_object_with_random_color(image_path, translation_range):
    image = Image.open(image_path)

    image_array = np.array(image)

    width, height = image.size

    background_color = find_background_color(image_path)

    translation_x = random.randint(-translation_range[0], translation_range[1])
    translation_y = random.randint(-translation_range[0], translation_range[1])

    translated_image = Image.new('RGB', (width, height), background_color)

    new_x = translation_x
    new_y = translation_y

    translated_image.paste(image, (new_x, new_y))

    return translated_image

def process_folder(input_folder, output_folder, translation_range):
    for subfolder in os.listdir(input_folder):
        subfolder_input_path = os.path.join(input_folder, subfolder)
        subfolder_output_path = os.path.join(output_folder, subfolder)

        if not os.path.exists(subfolder_output_path):
            os.makedirs(subfolder_output_path)

        for filename in os.listdir(subfolder_input_path):
            if filename.endswith(".JPEG") :
                image_path = os.path.join(subfolder_input_path, filename)
                translated_image = translate_object_with_random_color(image_path, translation_range)
                output_path = os.path.join(subfolder_output_path, filename)
                translated_image.save(output_path)

if __name__ == "__main__":
    input_folder = "D:/study/msc/project/feedback-attention-cnn-main/dataset/Test_name"
    output_folder = "D:/study/msc/project/feedback-attention-cnn-main/dataset/test_location"
    translation_range = (100, 100)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_folder(input_folder, output_folder, translation_range)
