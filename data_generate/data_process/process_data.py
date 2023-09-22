import numpy as np
import pickle as pk
import os
import cv2
import mmcv

def read_pickle(fname):
    with open(fname, 'rb') as f:
        context = pk.load(f)
    return context


def write_pickle(fname, datasets):
    with open(fname, 'wb') as f:
        pk.dump(datasets, f)


# (1,31,17,2), (1,31,17)
def reformat_keypoints(datasets):
    annos = datasets['annotations']
    new_kp = [0, 5, 6, 7, 8, 9, 10, 11, 12]
    for anno in annos:
        keypoint = anno['keypoint']
        keypoint_score = anno['keypoint_score']
        
        lshoulder = keypoint[..., 5, :]
        rshoulder = keypoint[..., 6, :]
        lhip = keypoint[..., 11, :]
        rhip = keypoint[..., 12, :]
        neck = 1/2 * (lshoulder + rshoulder)
        midhip = 1/2 * (lhip + rhip)
        
        keypoint = keypoint[..., new_kp, :]
        keypoint = np.concatenate(
            (keypoint, neck[..., np.newaxis, :], midhip[..., np.newaxis, :]),
            axis=2
        )
        # keypoint[..., 9, :] = neck[:, np.newaxis, :]
        # keypoint[..., 10, :] = midhip[:, np.newaxis, :]
        

        lshoulder_score = keypoint_score[..., 5]
        rshoulder_score = keypoint_score[..., 6]
        lhip_score = keypoint_score[..., 11]
        rhip_score = keypoint_score[..., 12]
        neck_score = 1/2 * (lshoulder_score + rshoulder_score)
        midhip_score = 1/2 * (lhip_score + rhip_score)
        
        keypoint_score = keypoint_score[..., new_kp]
        keypoint_score = np.concatenate(
            (keypoint_score, neck_score[..., np.newaxis], midhip_score[..., np.newaxis]),
            axis=2
        )
        
        anno['keypoint'] = keypoint
        anno['keypoint_score'] = keypoint_score
        
    return datasets


def update_confidence(datasets):
    annos = datasets['annotations']
    for anno in annos:
        keypoint_score = anno['keypoint_score']
        anno['keypoint_score'] = np.ones_like(keypoint_score)

    return datasets 


def verify_error(fname_gt, fname_pred):
    with open(fname_gt, 'rb') as f:
        gt = pk.load(f)

    with open(fname_pred, 'rb') as f:
        pred = pk.load(f)        
    

    true_num_ = 0
    true_num = 0
    true_uncert = []
    false_uncert = []
    error_name = []
    
    for idx, item in enumerate(gt['split']['test']):
        
        # 根据gt中sub_val的划分，获取真值
        for i in gt['annotations']:
            if item == i['frame_dir']:
                gt_label = i['label']
                break
            
        # 单个模型的预测结果
        pred_label = np.argmax(pred[idx])
        if gt_label == pred_label:         
            true_num += 1
        else:
            error_name.append(item[-9:-5])

    test_num = len(gt['split']['test'])
    print(error_name)
    # print(true_num/test_num)
    # print(len(error_name))


def remove_datasets(dirname):
    rm_list = ['0110', '0086', '0013', '0125', '0041', '0143', '0155', '0107', '0005', '0083', '0025', '0074', '0068', '0137', '0053', '0104', '0080', '0062', '0149', '0173', '0029', '0017', '0101', '0113', '0131', '0047']
    files = os.listdir(dirname)
    
    # 删除不符合的数据
    for file in files:
        if file[-13:-9] in rm_list:
            # print(file)
            os.remove(os.path.join(dirname, file))
            
    # # 筛选测试集
    # test_set = []
    # for file in files:     
    #     if file[-8:-4] == 'test':
    #         test_set.append(file)
    
    # print(len(test_set))
    # def custom_sort(file_name):
    #     try:
    #         number = int(file_name[-13:-9])
    #         return number
    #     except ValueError:
    #         return float('inf')
        
    # test_set = sorted(test_set, key=custom_sort)
    # print(len(test_set))
    
    # for idx, file in enumerate(test_set):
    #     if idx <= 115:
    #         continue
    #     os.remove(os.path.join(dirname, file))
    

def get_action_num(dirname, action):
    files = os.listdir(dirname)
    action_num = 0
    for file in files:
        if action in file and 'test' in file:
            action_num += 1
    print(action_num)
            
def remove_putdown(dirname):
    files = os.listdir(dirname)
    for file in files:
        if 'putdown' in file:
            os.remove(os.path.join(dirname, file))
       

def relabel_file(dirname):
    files = os.listdir(dirname)
    files = sorted(files)
    for idx, file in enumerate(files):
        fstring = file.split('_')
        fstring[2] = f"{idx:04}"
        new_file = "_".join(fstring)
        print(new_file)
        os.rename(os.path.join(dirname, file), os.path.join(dirname, new_file))

    
if __name__ == '__main__':
    
    fname = 'data_generate/datasets_action_5_10fps_120test/custom_hrnet.pkl'
    outfname = 'data_generate/datasets_action_5_10fps_120test/custom_hrnet_11kp.pkl'
    context = read_pickle(fname)
    datasets = reformat_keypoints(context)
    write_pickle(outfname, datasets)
    
    context = read_pickle('data_generate/datasets_action_5_10fps_120test/custom_hrnet_11kp.pkl')
    datasets = update_confidence(context)
    write_pickle('data_generate/datasets_action_5_10fps_120test/custom_hrnet_11kp_1conf.pkl', context)
    
    
    # read_pickle('result.pkl')
    # verify_error('data_generate/datasets_action_5_10fps_120test/custom_hrnet_11kp_1conf.pkl', 'result.pkl')
    
    # remove_datasets("data_generate/datasets_action_6_10fps_test120/videos__")
    # print(len(os.listdir("data_generate/datasets_action_6_10fps_test120/videos__")))
    # print(len(os.listdir("data_generate/datasets_action_6/videos__")))
    # remove_datasets("data_generate/datasets_action_6_10fps_test120/videos__")
    # get_action_num("data_generate/datasets_action_6_10fps_test120/videos__", "right")
    
    
    # remove_putdown("data_generate/datasets_action_5_10fps_120test/videos__")
    # print(len(os.listdir("data_generate/datasets_action_5_10fps_120test/videos__")))
    
    # relabel_file("data_generate/datasets_action_6_10fps_test/tttt_")
    
    # relabel_file("data_generate/datasets_action_5_10fps_120test/videos__")
    # print(len(os.listdir("data_generate/datasets_action_5_10fps_120test/videos_train")))
    # print(len(os.listdir("data_generate/datasets_action_6_10fps_test/test")))
    

    
    # relabel_file("data_generate/datasets_action_5_10fps_120test/videos_test")
    # print(len(os.listdir("data_generate/datasets_action_5_10fps_120test/videos_all")))
    # print(len(os.listdir("data_generate/datasets_action_5_10fps_120test/videos_test_update")))
    
    
    # print(len(os.listdir("data_generate/datasets_action_5_10fps_120test/videos_all")))
    
    
    
    # 用73，90 替换 83，85
    # right_person1_0073_test
    # right_person1_0090_test
    
    # right_person1_0083_test
    # right_person1_0085_test
    
    # with open('data_generate/datasets_action_5_10fps_120test_old/custom_annos.pkl', 'rb') as f:
    #     src = pk.load(f)
    
    # with open('data_generate/datasets_action_5_10fps_120test/custom_annos.pkl', 'rb') as f:
    #     dst = pk.load(f)
      
    # for idx, s in enumerate(src):
    #     if s['frame_dir'] == "right_person1_0073_test":
    #         select = s
    #         break     
    # for idx, d in enumerate(dst):
    #     if d['frame_dir'] == "right_person1_0083_test":
    #         select['frame_dir'] = d['frame_dir']
    #         dst[idx] = select
    #         break
        
    
    # for idx, s in enumerate(src):
    #     if s['frame_dir'] == "right_person1_0090_test":
    #         select = s
    #         break
    # for idx, d in enumerate(dst):
    #     if d['frame_dir'] == "right_person1_0085_test":
    #         print(select['frame_dir'])
    #         select['frame_dir'] = d['frame_dir']
    #         dst[idx] = select
    #         print(dst[idx]['frame_dir'])
    #         break
        
    # with open('data_generate/datasets_action_5_10fps_120test/custom_annos_.pkl', 'wb') as f:
    #     pk.dump(dst, f)
    
    
    