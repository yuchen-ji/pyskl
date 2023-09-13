import numpy as np
import pickle as pk

ensemble_num = 4
pred_ensemble = []
for idx in range(ensemble_num):
    with open(f'/workspaces/pyskl/work_dirs/stgcn_ntu60_coco_{idx+1}/last_pred.pkl', 'rb') as f:
        pred = pk.load(f)
    pred_ensemble.append(pred)



with open('/workspaces/pyskl/data_generate/datasets_ntu60/ntu60_hrnet.pkl', 'rb') as f:
    gt = pk.load(f)
print(gt.keys())



true_num = 0
for idx, item in enumerate(gt['split']['xsub_val']):
    
    # 根据gt中sub_val的划分，获取真值
    for i in gt['annotations']:
        if item == i['frame_dir']:
            gt_label = i['label']
            break
            
    all = np.concatenate(
        ([pred_ensemble[i][idx][np.newaxis,...] for i in range(ensemble_num)]),
        axis = 0
    )
    
    pred = np.argmax(np.mean(all, axis=0))
    pred_label = np.argmax(pred)
    
    if gt_label == pred_label:
        true_num += 1

print(true_num)