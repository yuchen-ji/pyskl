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
    action = ['stand', 'left', 'right', 'polish', 'assembly', 'putdown']
    
    if split == 'train':
        remainder = index % 18
        if remainder in [0, 1, 2]:
            name = action[0]
        if remainder in [3, 4, 5]:
            name = action[1]
        if remainder in [6, 7, 8]:
            name = action[2]
        if remainder in [9, 10, 11]:
            name = action[3]
        if remainder in [12, 13, 14]:
            name = action[4]
        if remainder in [15, 16, 17]:
            name = action[5]        
    
    if split == 'test':
        remainder = index % 6
        if remainder in [0]:
            name = action[0]
        if remainder in [1]:
            name = action[1]
        if remainder in [2]:
            name = action[2]
        if remainder in [3]:
            name = action[3]
        if remainder in [4]:
            name = action[4]
        if remainder in [5]:
            name = action[5]

    return name


def rename_files(file_names, file_path, split, start_idx, end_idx, start):
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
        
        new_name = f"{action}_person1_{index + start:04}_{split}.MOV"      
        dst_path = os.path.join(file_path, new_name)
        src_path = os.path.join(file_path, file_name)
        os.rename(src_path, dst_path)
     

if __name__ == "__main__":
        
    # file_path = "data_generate/datasets_action_6/videos"
    file_path = "data_generate/video_action6_test"
    split = "test"
    
    # start = 0
    # start_idx = 2994
    # end_idx = 3105
    
    # 然后手动标注 3组 right， polish， assembly putdown
    
    # start = 108
    # start_idx = 3115
    # end_idx = 3224
    
    # start = 216
    # start_idx = 3229
    # end_idx = 3446
    
    # start = 431
    # start_idx = 3449
    # end_idx = 3666
    
    # start = 0
    # start_idx = 3667
    # end_idx = 3685
    
    # By yuchen 23.09.13 
    start = 0
    start_idx = 3776
    end_idx = 3965
    
    sort_names = read_sort_files(file_path)
    rename_files(sort_names, file_path, split, start_idx, end_idx, start)