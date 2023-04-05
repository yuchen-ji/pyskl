import requests
import pandas as pd
import pickle as pk
import os, sys
import numpy as np

import seaborn as sns

sys.path.insert(0, '/workspaces/pyskl/workspace')

def select_kp():
    with open(r'workspace/ntu60_hrnet.pkl', 'rb') as f:
        context = pk.load(f)
        sub_kp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        for each in context['annotations']:
            each['keypoint'] = each['keypoint'][...,sub_kp,:]
            each['keypoint_score'] = each['keypoint_score'][...,sub_kp,:]     
    with open(r'workspace/diving48_hrnet_sub.pkl', 'wb') as f:
        pk.dump(context, f)
    print(context['annotations'][0]['keypoint'].shape)
    

def select_classes():
    with open(r'workspace/data/ntu60_hrnet.pkl', 'rb') as f:
        context = pk.load(f)

    # select_label= [i for i in range(60)]
    # select_label = [6, 18, 23, 31, 38]
    select_label = [6, 8, 9, 27, 31, 34]
    select_label = [idx - 1 for idx in select_label]
    
    # remove_label = [10, 20, 30, 40, 50]
    # for each in remove_label:
    #     select_label.remove(each)
        
    select_name = []
    select_annos = []
    # select anno
    for each in context['annotations']:
        if each['label'] in select_label:
            select_name.append(each['frame_dir'])
            select_annos.append(each)
            # rename labels
            each['label'] = select_label.index(each['label'])
    context['annotations'] = select_annos
    
    # select split
    for item in range(len(context['split']['xsub_train'])-1, -1, -1):
        if context['split']['xsub_train'][item] not in set(select_name):
            del context['split']['xsub_train'][item]
    for item in range(len(context['split']['xsub_val'])-1, -1, -1):
        if context['split']['xsub_val'][item] not in set(select_name):
            del context['split']['xsub_val'][item]

    # dump
    with open(r'workspace/data/nuaa6.pkl', 'wb') as f:
        pk.dump(context, f)
    print(len(context['split']['xsub_train']))
        

def calc_uncertainty():
    ensemble = []
    for i in range(1, 5, 1):
        with open(r'workspace/results/real_time/result_{}.pkl'.format(i), 'rb') as f:
            context = pk.load(f)
            ensemble.append(context)
    
    # calculate uncertainty
    entropy_list = []
    prediction_list = []
    uncertainty_list = []
    
    for item in range(len(ensemble[0])):
            
        all = np.concatenate((ensemble[0][item][np.newaxis,...],
                              ensemble[1][item][np.newaxis,...],
                              ensemble[2][item][np.newaxis,...],
                              ensemble[3][item][np.newaxis,...],
                            #   ensemble[4][item][np.newaxis,...]
                              ), axis=0)
        pred_entropy = -np.sum(
            np.mean(all, axis=0) * np.log(np.mean(all, axis=0)) + sys.float_info.min,
            axis=-1
        )
        
        # disagreements / uncertainty
        pred_mean = np.mean(all, axis=0)
        uncertainty = 0
        for each in all:
            uncertainty += np.sum(each * (np.log(each+1e-20) - np.log(pred_mean+1e-20)))
            
        entropy_list.append(pred_entropy)
        prediction_list.append(max(pred_mean))
        uncertainty_list.append(uncertainty)
        
    entropy_list = np.array(entropy_list)
    prediction_list = np.array(prediction_list)
    uncertainty_list = np.array(uncertainty_list)
    
    exp = {
            'entropy_list':entropy_list,
            'prediction_list':prediction_list,
            'uncertainty_list':uncertainty_list,
           }
    
    # sort = np.sort(uncertainty_list, axis=0, kind='mergesort')
    with open(r'workspace/results/real_time/all_uncertainty.pkl', 'wb') as f:
        pk.dump(exp, f)
    
    print(f"mean: {np.mean(entropy_list)}")
    print(f"std: {np.std(entropy_list)}")
    print(f"max: {np.max(entropy_list)}")
    print(f"min: {np.min(entropy_list)}")
    print(len(entropy_list))
    

def verify_error():
    ensemble = []
    for i in range(1, 5, 1):
        with open(r'workspace/results/nuaa6/result_{}.pkl'.format(i), 'rb') as f:
            context = pk.load(f)
            ensemble.append(context)
    with open(r'workspace/data/nuaa6.pkl', 'rb') as f:
        gt = pk.load(f)
    
    error = []
    error_uncertainty = []
    true_uncertainty = []
    for item in range(len(ensemble[0])):
        all = np.concatenate((ensemble[0][item][np.newaxis,...],
                              ensemble[1][item][np.newaxis,...],
                              ensemble[2][item][np.newaxis,...],
                              ensemble[3][item][np.newaxis,...],
                             ), axis=0)
        
        prediction = np.argmax(np.mean(all, axis=0))
        pred_value = np.max(np.mean(all, axis=0))
        gt_name = gt['split']['xsub_val'][item]
        for each in gt['annotations']:
            if each['frame_dir'] == gt_name:
                label = each['label']
                # prediction
                entropy = -np.sum(
                    np.mean(all, axis=0) * np.log(np.mean(all, axis=0)) + sys.float_info.min,
                    axis=-1
                )
                # disagreements / uncertainty
                pred_mean = np.mean(all, axis=0)
                uncertainty = 0
                
                for each in all:
                    uncertainty += np.sum(each * (np.log(each+1e-20) - np.log(pred_mean+1e-20)))
                if prediction != label:                    
                    # error.append(item)
                    error.append((item, pred_value, uncertainty))
                    error_uncertainty.append(uncertainty)
                else:
                    true_uncertainty.append(uncertainty)                                       
                break
    print(error)
    print(len(error_uncertainty))
    
    error_uncertainty = np.array(error_uncertainty)
    with open(r'workspace/results/nuaa6/error_uncertainty.pkl', 'wb') as f:
        pk.dump(error_uncertainty, f)
    with open(r'workspace/results/nuaa6/true_uncertainty.pkl', 'wb') as f:
        pk.dump(true_uncertainty, f)


def read_pickle(filename):
    with open(filename, 'rb') as f:        
        context = pk.load(f)
    # for e in context['annotations']:
    #     print(e['keypoint'].shape[2])
    return context


# 将数据集划分成不同的patch，用来重命名不同的patch中对应的label，这个目前是不用了
def rename_label_index():
    for i in range(1, 6, 1):
        with open(r'workspace/data/ntu60_exc{}.pkl'.format(i), 'rb') as f:
            context = pk.load(f)
        label_list = [1, 2, 3, 4, 5]
        label_list.remove(i)
            
        for each in context['annotations']:
            each['label'] = label_list.index(each['label']+1)

        with open(r'workspace/data/ntu5_sub{}.pkl'.format(i), 'wb') as f:
            pk.dump(context, f)            


# 这一段用来重定义keypoints的得分，将其全设为1
def rescore_keypoints():
    filename='workspace/data/nuaa6.pkl'
    with open(filename, 'rb') as f:        
        context = pk.load(f)
    for anno in context['annotations']:
        frames = anno['total_frames']
        keypoint_num = 17
        keypoints_score = np.ones((1, frames, keypoint_num), dtype=np.float16)
        anno['keypoint_score'] = keypoints_score

    with open('workspace/data/nuaa6_score.pkl', 'wb') as f:
        pk.dump(context, f)


def concat_pkl():
    dataset = {}
    annotations = []
    split = []
    dirname = 'workspace/data/annotation'
    for idx, file in enumerate(os.listdir(dirname)):
        with open(os.path.join(dirname, file), 'rb') as f:
            context = pk.load(f)
        
        keypoints = context['keypoint']
        scores = context['keypoint_score']
        
        # 忽略后四个关节，这是身体的下半部分
        # keypoints = keypoints[:, :, :-4, :]
        # scores = scores[:, :, :-4]
        
        context['keypoint'] = keypoints
        context['keypoint_score'] = scores   
        context['frame_dir'] = 'factory_{}'.format(idx)     
        annotations.append(context)
        split.append(context['frame_dir'])
    dataset['split'] = {'xsub_train': split}
    dataset['annotations'] = annotations
    
    with open(r'workspace/data/factory6_coco.pkl', 'wb') as f:
        pk.dump(dataset, f)
    

if __name__ == '__main__':
    
    context = read_pickle(filename='workspace/data/factory6_coco.pkl')
    # select_classes()
    # rename_label_index()
    # calc_uncertainty()
    # verify_error()
    # rescore_keypoints()
    # concat_pkl()