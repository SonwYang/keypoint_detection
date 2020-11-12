from datasets.base import BaseDataset
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
# from .DOTA_devkit.ResultMerge_multi_process import mergebypoly


def custom_blur_demo(image):
  kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
  dst = cv2.filter2D(image, -1, kernel=kernel)
  return dst


class DOTA(BaseDataset):
    def __init__(self, data_dir, phase, input_h=512, input_w=512, down_ratio=4):
        super(DOTA, self).__init__(data_dir, phase, input_h, input_w, down_ratio)
        self.category = [
                        '1'
                         ]
        self.color_pans = [(204,78,210),
                           (0,192,255),
                           (0,131,0),
                           (240,176,0),
                           (254,100,38),
                           (0,0,255),
                           (182,117,46),
                           (185,60,129),
                           (204,153,255),
                           (80,208,146),
                           (0,0,204),
                           (17,90,197),
                           (0,255,255),
                           (102,255,102),
                           (255,255,0)]
        self.num_classes = len(self.category)
        self.cat_ids = {cat:i for i,cat in enumerate(self.category)}
        self.img_ids = self.load_img_ids()
        self.image_path = os.path.join(data_dir, 'images')
        self.label_path = os.path.join(data_dir, 'labelTxt')

    def load_img_ids(self):
        if self.phase == 'train':
            image_set_index_file = os.path.join(self.data_dir, 'train.txt')
        else:
            image_set_index_file = os.path.join(self.data_dir, 'valid.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file, 'r') as f:
            lines = f.readlines()
        image_lists = [line.strip() for line in lines]
        return image_lists

    def letterbox(self, img, new_shape, color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def load_image(self, index, only_img=True):
        img_id = self.img_ids[index]
        imgFile = os.path.join(self.image_path, img_id+'.jpg')
        assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
        img = cv2.imread(imgFile)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        new_img, ratio, (dw, dh) = self.letterbox(img.copy(), new_shape=[self.input_h, self.input_w], auto=False, scaleup=False)
        # img = custom_blur_demo(img)
        # if random.random() > 0.5:
        #     img = self.RA(img)
        if only_img:
            return new_img
        else:
            return new_img, ratio, (dw, dh)

    def load_annoFolder(self, img_id):
        return os.path.join(self.label_path, img_id+'.txt')

    def load_annotation(self, index):
        image, ratio, pad = self.load_image(index, only_img=False)
        valid_pts = []
        valid_cat = []
        with open(self.load_annoFolder(self.img_ids[index]), 'r') as f:
            for i, line in enumerate(f.readlines()):
                obj = line.split(' ')  # list object
                # if len(obj) > 8:
                x1 = float(obj[0])
                y1 = float(obj[1])
                x2 = float(obj[2])
                y2 = float(obj[3])
                # cls = self.cat_ids[obj[4]]
                cls = 0

                valid_pts.append([x1, y1, x2, y2])
                valid_cat.append(cls)
        f.close()

        valid_pts = np.asarray(valid_pts, np.float32)
        new_boxes = np.zeros_like(valid_pts)
        new_boxes[:, 0] = ratio[0] * valid_pts[:, 0] + pad[0]  # pad width
        new_boxes[:, 1] = ratio[1] * valid_pts[:, 1] + pad[1]  # pad height
        new_boxes[:, 2] = ratio[0] * valid_pts[:, 2] + pad[0]
        new_boxes[:, 3] = ratio[1] * valid_pts[:, 3] + pad[1]
        new_boxes = np.clip(new_boxes, a_min=0, a_max=self.input_w - 1)

        #### cv_show
        # for box in new_boxes:
        #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
        # plt.imshow(image)
        # plt.show()

        annotation = {}
        annotation['pts'] = new_boxes
        annotation['cat'] = np.asarray(valid_cat, np.int32)
        return annotation


def collater(data):
    out_data_dict = {}
    for name in data[0]:
        out_data_dict[name] = []
    for sample in data:
        for name in sample:
            out_data_dict[name].append(torch.from_numpy(sample[name]))
    for name in out_data_dict:
        out_data_dict[name] = torch.stack(out_data_dict[name], dim=0)
    return out_data_dict

if __name__ == '__main__':
    import torch
    dataset = DOTA(data_dir='J:/dl_dataset/object_detection/aircraft',
                   phase='train',
                   input_h=512,
                   input_w=512,
                   down_ratio=4)
    dataloader = torch.utils.data.DataLoader(dataset,
                                           batch_size=2,
                                           shuffle=True,
                                           num_workers=0,
                                           pin_memory=True,
                                           drop_last=True,
                                           collate_fn=collater)
    for data_dict in dataloader:
        print(data_dict['input'].shape,
              data_dict['hm'].shape,
              data_dict['reg_mask'].shape)
