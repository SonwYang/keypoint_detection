import torch.utils.data as data
import cv2
import torch
import numpy as np
import math
from .draw_gaussian import draw_umich_gaussian, gaussian_radius
from .transforms import random_flip, load_affine_matrix, random_crop_info, ex_box_jaccard
import random
from .randaugment import RandomAugment
from .transfroms_od import *


class BaseDataset(data.Dataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None):
        super(BaseDataset, self).__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.img_ids = None
        self.num_classes = None
        self.max_objs = 500
        self.transforms = DualCompose([
                                            CoarseDropout(p=1.),
                                            RandomRotate90(p=0.5),
                                            RandomFlip(p=0.5)
                                        ])
        self.RA = RandomAugment()

    def load_img_ids(self):
        return None

    def load_image(self, index, only_img=True):
        return None

    def load_annoFolder(self, img_id):
        return None

    def load_annotation(self, index):
        return None

    def dec_evaluation(self, result_path):
        return None

    def __len__(self):
        return len(self.img_ids)

    def processing_test(self, image, input_h, input_w):
        image = cv2.resize(image, (input_w, input_h))
        out_image = image.astype(np.float32) / 255.
        out_image = out_image - 0.5
        out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
        out_image = torch.from_numpy(out_image)
        return out_image

    def generate_ground_truth(self, image, annotation):
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        image = np.transpose(image / 255. - 0.5, (2, 0, 1))

        image_h = self.input_h // self.down_ratio
        image_w = self.input_w // self.down_ratio

        hm = np.zeros((self.num_classes, image_h, image_w), dtype=np.float32)
        ## add end
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        num_objs = min(annotation['pts'].shape[0], self.max_objs)
        # ###################################### view Images #######################################
        # copy_image1 = cv2.resize(image, (image_w, image_h))
        # copy_image2 = copy_image1.copy()
        # ##########################################################################################
        for k in range(num_objs):
            rect = annotation['pts'][k, :] // self.down_ratio
            x1, y1, x2, y2 = rect
            cen_x, cen_y, bbox_w, bbox_h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            # print(theta)
            radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
            radius = max(0, int(radius))
            ct = np.asarray([cen_x, cen_y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm[annotation['cat'][k]], ct_int, radius)
            ind[k] = ct_int[1] * image_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1

        ret = {'input': image,
               'hm': hm,
               'reg_mask': reg_mask,
               'ind': ind,
               'reg': reg,
               }
        return ret

    def __getitem__(self, index):
        if self.phase == 'test':
            image = self.load_image(index)
            image_h, image_w, c = image.shape
            img_id = self.img_ids[index]
            image = self.processing_test(image, self.input_h, self.input_w)
            return {'image': image,
                    'img_id': img_id,
                    'image_w': image_w,
                    'image_h': image_h}
        else:
            # image, annotation = self.load_img_and_ann(index)
            if random.random() > 0.5:
                image, annotation = self.load_img_and_ann(index)
            else:
                image, annotation = self.load_mixup_img_and_ann(index)
            # data augmemtations in color
            image = image.astype(np.uint8)
            augment_hsv(image)
            # data augmentations in geometry
            image, annotation['pts'] = self.transforms(image, annotation['pts'])
            data_dict = self.generate_ground_truth(image, annotation)
            return data_dict

    def load_img_and_ann(self, index):
        image = self.load_image(index)
        annotation = self.load_annotation(index)
        return image, annotation

    def load_mixup_img_and_ann(self, index):
        image, annotation = self.load_img_and_ann(index)
        image_r, annotation_r = self.load_img_and_ann(random.randint(0, len(self.img_ids) - 1))
        for k, v in annotation.items():
            annotation[k] = np.array(v.tolist() + annotation_r[k].tolist())
            # annotation[k] = np.stack((v[0], annotation_r[k][0]), axis=1)
        return (image + image_r) / 2, annotation


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed