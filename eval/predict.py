from config_eval import Config
from models import ctrbox_net
import decoder
import torch
import glob
import tqdm
import os
import cv2
import numpy as np


def load_net(cfg):
    heads = {
             'hm': cfg.num_classes,
             'reg': 2,
             }
    down_ratio = 4
    model = ctrbox_net.CTRBOX(heads=heads,
                               pretrained=True,
                               down_ratio=down_ratio,
                               final_kernel=1,
                               head_conv=32)
    checkpoint = torch.load(cfg.resume, map_location=lambda storage, loc: storage)
    print('loaded weights from {}, epoch {}'.format(cfg.resume, checkpoint['epoch']))
    state_dict_ = checkpoint['model_state_dict']
    model.load_state_dict(state_dict_, strict=True)
    return model


def decode_prediction(predictions, category, down_ratio):
    predictions = predictions[0, :, :]

    pts0 = []
    scores0 = []
    for pred in predictions:
        score = pred[-2]
        pred[:2] = pred[:2] * down_ratio
        xmin = pred[0] - 10
        ymin = pred[1] - 10
        xmax = pred[0] + 10
        ymax = pred[1] + 10
        clse = pred[-1]
        pts = np.asarray([xmin, ymin, xmax, ymax, clse], np.float32)
        pts0.append(pts)
        scores0.append(score)
    return pts0, scores0


def make_predictions(net, subImg, cfg):
    with torch.no_grad():
        pr_decs = net(subImg)
    torch.cuda.synchronize(device)
    predictions = decoder.ctdet_decode(pr_decs)
    pts0, scores0 = decode_prediction(predictions, cfg.class_dict, cfg.down_ratio)
    return pts0, scores0


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
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


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = Config()
    mkdir(cfg.out_dir)
    txtPath = os.path.join(cfg.data_root, 'valid.txt')
    imgList = []
    f = open(txtPath)
    datas = f.readlines()
    for data in datas:
        path = data.split(' ')[0]
        imgList.append(path)

    imgList = [os.path.join(cfg.data_root, 'images', f'{imgName}.jpg') for imgName in imgList]
    a = imgList.sort()
    net = load_net(cfg).to(device)
    net.eval()
    decoder = decoder.DecDecoder(K=cfg.K,
                                 conf_thresh=cfg.conf_thresh,
                                 num_classes=cfg.num_classes)

    frame_size = cfg.image_size - cfg.gap
    for j, imgPath in tqdm.tqdm(enumerate(imgList)):
        image_name = os.path.split(imgPath)[-1].split('.')[0]
        image = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img2, ratio, pad = letterbox(image.copy(), (cfg.image_size, cfg.image_size), auto=False, scaleup=False)
        sample = img2.copy()
        img2 = img2.astype(np.float32) / 255.
        img2 -= 0.5
        img2 = img2.transpose(2, 0, 1).reshape(1, 3, cfg.image_size, cfg.image_size)
        img2 = torch.from_numpy(img2).to(device)
        bboxes, scores = make_predictions(net, img2, cfg)
        picked_boxes, picked_score = nms(bboxes, scores, threshold=0.3)
        for box, score in zip(picked_boxes, picked_score):
            # cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
            center = (int((box[2] + box[0])/2), int((box[3] + box[1])/2))
            cv2.circle(sample, center, 4, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(cfg.out_dir, f"{image_name}.jpg"), sample)




