class Config(object):
    # data_root = 'F:/dl_dataset/l4/test_big'
    data_root = 'J:/dl_dataset/object_detection/aircraft'
    # data_root = 'F:/dl_dataset/l4/train/images'
    # data_root = 'E:/2_1/hrsc/images'
    # data_root = 'E:/2_1/step2_BBAVectors/eval/test'
    out_dir = 'predict_result'
    resume = 'F:/key_point/centernet/weights_dota/model_26.pth'
    class_dict = ['1']
    num_classes = len(class_dict)
    image_size = 1024
    gap = image_size // 2
    K = 128    # maximum of objects
    conf_thresh = 0.6
    down_ratio = 4
    tta = True
    multi_scale = True
    scales = [1.0, 0.5, 0.25]


if __name__ == '__main__':
    import numpy as np
    arr = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
           [5, 6, 7, 8, 1, 2, 3, 4]])
    print(arr[:, [0, 2, 4, 6]])
    print(arr.dtype)
    b = arr.tolist()
