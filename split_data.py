import os
import random
import shutil

def split_train_test(input_folder, output_folder, test_ratio=0.2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    subfolders = [subfolder for subfolder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, subfolder))]
    for subfolder in subfolders:
        subfolder_input_path = os.path.join(input_folder, subfolder)
        subfolder_output_path = os.path.join(output_folder, subfolder)

        if not os.path.exists(subfolder_output_path):
            os.makedirs(subfolder_output_path)

        file_list = os.listdir(subfolder_input_path)

        num_files = len(file_list)
        num_test_files = int(num_files * test_ratio)

        test_files = random.sample(file_list, num_test_files)

        for filename in test_files:
            src_path = os.path.join(subfolder_input_path, filename)
            dest_path = os.path.join(subfolder_output_path, filename)
            shutil.move(src_path, dest_path)

if __name__ == "__main__":
    input_folder = "/nobackup/sc22x2l/feedback_attention/train/"
    output_folder = "/nobackup/sc22x2l/feedback_attention/test/"

    split_train_test(input_folder, output_folder, test_ratio=0.2)
