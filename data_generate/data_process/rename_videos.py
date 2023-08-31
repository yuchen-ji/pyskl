import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def read_sort_files(dir_name):
    file_names = os.listdir(dir_name)
    def custom_sort(file_name):
        try:
            number = int(file_name[-8:-4])
            return number
        except ValueError:
            return float('inf')
        
    sorted_names = sorted(file_names, key=custom_sort)
    return sorted(sorted_names)


def get_action(index, split):
    name = 'NULL'
    action = ['stand', 'left', 'right']
    if split == 'train':
        if index % 6 == 0 or index % 6 == 1:
            name = action[0]
        if index % 6 == 2 or index % 6 == 3:
            name = action[1]
        if index % 6 == 4 or index % 6 == 5:
            name = action[2]
    
    if split == 'test':
        if index % 3 == 0:
            name = action[0]
        if index % 3 == 1:
            name = action[1]
        if index % 3 == 2:
            name = action[2]
    
    return name


def rename_files(file_names, file_path, split, start_idx, end_idx):
    for index, file_name in enumerate(file_names):
        try:
            if int(file_name[-8:-4]) < start_idx:
                continue
            if int(file_name[-8:-4]) > end_idx:
                break
        except ValueError:
            break
        
        # 获取动作名称
        action = get_action(index, split)
        
        new_name = f"{action}_person1_{index:04}_{split}.mp4"      
        dst_path = os.path.join(file_path, new_name)
        src_path = os.path.join(file_path, file_name)
        os.rename(src_path, dst_path)
     

if __name__ == "__main__":
        
    file_path = "data_generate/videos__"
    split = "test"
    start_idx = 2970
    end_idx = 2980
    
    sort_names = read_sort_files(file_path)
    rename_files(sort_names, file_path, split, start_idx, end_idx)