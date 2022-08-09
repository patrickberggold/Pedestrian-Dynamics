import os
import random

def get_filenames(path, extension):
        files_list = list()
        folder_list = sorted(os.listdir(path), key=lambda x: int(x.split('__')[0]))
        for foldername in folder_list:
            files_in_folder = sorted(os.listdir(os.path.join(path, foldername)), key=lambda x: int(x.split('_')[-1].replace(extension, '')))
            for filename in files_in_folder:
                if filename.endswith(extension):
                    files_list.append(os.path.join(path, foldername, filename))
        return files_list

def get_flooplan_simple_split(csv_path, img_path):
    splits = [0.7, 0.15, 0.15]
    train_split_factor, val_split_factor, test_split_factor = splits

    assert os.path.isdir(csv_path)
    assert os.path.isdir(img_path)
    all_txt_files = get_filenames(csv_path, '.txt')
    all_img_files = get_filenames(img_path, '.h5')

    assert len(all_txt_files)==len(all_img_files)

    len_dataset = len(all_txt_files)
    indices = list(range(len_dataset))

    random.shuffle(indices)

    all_txt_files = [all_txt_files[i] for i in indices]
    all_img_files = [all_img_files[i] for i in indices]

    val_split_index = int(len_dataset * val_split_factor)
    test_split_index = int(len_dataset * test_split_factor)
    
    train_txt_list = all_txt_files[(test_split_index + val_split_index):]
    val_txt_list = all_txt_files[test_split_index:(test_split_index + val_split_index)]
    test_txt_list = all_txt_files[:test_split_index]

    train_img_list = all_img_files[(test_split_index + val_split_index):]
    val_img_list = all_img_files[test_split_index:(test_split_index + val_split_index)]
    test_img_list = all_img_files[:test_split_index]

    return [train_txt_list, val_txt_list, test_txt_list], [train_img_list, val_img_list, test_img_list]


