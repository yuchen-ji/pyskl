import copy
import numpy as np
# from operator import itemgetter


def slide_window(fake_anno, keypoint, keypoints_score, start_frame):
    # start_frame = 20
    iter_frames = 1
    fake_list = []
    for item in range(start_frame, len(keypoint[0]), iter_frames):
        anno = copy.deepcopy(fake_anno)
        anno['total_frames'] = start_frame
        anno['keypoint'] = copy.deepcopy(keypoint[:,item-start_frame:item])
        anno['keypoint_score'] = copy.deepcopy(keypoints_score[:,item-start_frame:item])
        fake_list.append(anno)
    
    return fake_list


def get_pred_uncertainty(results):
    # results: (num_model, num_classes)
    mean = np.mean(results, axis=0)
    uncertainty = 0
    for pred in results:
        uncertainty += np.sum(pred * (np.log(pred + 1e-20) - np.log(mean + 1e-20)))
        
    return mean, uncertainty
    

if __name__ == '__main__':
    # results = [(5, 0.9977215), (0, 0.0011571808), (3, 0.0008513182), (2, 0.00015149107), (1, 6.916702e-05), (4, 4.941394e-05)]
    # results = sorted(results, key=lambda x: x[0])
    # results = [item[1] for item in results]
    # results = np.array(results)
    # print(results)

    results = np.array([[0.0011571808, 6.916702e-05, 0.00015149107, 0.0008513182, 4.941394e-05, 0.9977215],
                       [0.0011571808, 6.916702e-05, 0.00015149107, 0.0008513182, 4.941394e-05, 0.9977215]])
    calculate_uncertainty(results)