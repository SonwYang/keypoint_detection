import cv2
import numpy as np
import math
import random


class RandomFlip:
    def __init__(self, p=1):
        self.prob = p

    def bbox_hflip(self, bbox, shape):  # skipcq: PYL-W0613
        """Flip a bounding box horizontally around the y-axis.

        Args:
            bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.

        Returns:
            tuple: A bounding box `(x_min, y_min, x_max, y_max)`.

        """
        x_min, y_min, x_max, y_max = bbox
        return shape - x_max, y_min, shape - x_min, y_max

    def bbox_vflip(self, bbox, shape):  # skipcq: PYL-W0613
        """Flip a bounding box vertically around the x-axis.

        Args:
            bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.

        Returns:
            tuple: A bounding box `(x_min, y_min, x_max, y_max)`.

        """
        x_min, y_min, x_max, y_max = bbox
        return x_min, shape - y_max, x_max, shape - y_min

    def __call__(self, img, bbox=None):
        w, h = img.shape[:2]
        bbox = (bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3])
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if bbox is not None:
                if d == 0:
                    bbox = self.bbox_vflip(bbox, w)
                elif d == 1:
                    bbox = self.bbox_hflip(bbox, w)
                elif d == -1:
                    bbox = self.bbox_hflip(bbox, w)
                    bbox = self.bbox_vflip(bbox, w)
                else:
                    raise ValueError("Invalid d value {}. Valid values are -1, 0 and 1".format(d))

        bbox = np.asarray(bbox).transpose((1, 0))
        return img, bbox


class RandomRotate90:
    def __init__(self, p=1.):
        self.prob = p

    def __call__(self, img, bbox=None):
        w, h = img.shape[:2]
        bbox = (bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3])
        if random.random() < self.prob:
            factor = random.randint(0, 3)
            img = np.rot90(img, factor)
            if bbox is not None:
                bbox = self.bbox_rot90(bbox, factor, w)
        bbox = np.asarray(bbox).transpose((1, 0))
        return img, bbox

    def bbox_rot90(self, bbox, factor, shape):  # skipcq: PYL-W0613
        """Rotates a bounding box by 90 degrees CCW (see np.rot90)

        Args:
            bbox (tuple): A bounding box tuple (x_min, y_min, x_max, y_max).
            factor (int): Number of CCW rotations. Must be in set {0, 1, 2, 3} See np.rot90.
            rows (int): Image rows.
            cols (int): Image cols.

        Returns:
            tuple: A bounding box tuple (x_min, y_min, x_max, y_max).

        """
        if factor not in {0, 1, 2, 3}:
            raise ValueError("Parameter n must be in set {0, 1, 2, 3}")
        x_min, y_min, x_max, y_max = bbox[:4]
        if factor == 1:
            bbox = y_min, shape - x_max, y_max, shape - x_min
        elif factor == 2:
            bbox = shape - x_max, shape - y_max, shape - x_min, shape - y_min
        elif factor == 3:
            bbox = shape - y_max, x_min, shape - y_min, x_max
        return bbox


class Rotate:
    def __init__(self, limit=90, prob=0.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)
            height, width = img.shape[0:2]
            mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
            img = cv2.warpAffine(img, mat, (height, width),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpAffine(mask, mat, (height, width),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)

        return img, mask


class Cutout:
    def __init__(self, num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, prob=1.):
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value
        self.prob = prob

    def __call__(self, img, box=None):
        if random.random() < self.prob:
            h = img.shape[0]
            w = img.shape[1]
            # c = img.shape[2]
            # img2 = np.ones([h, w], np.float32)
            for _ in range(self.num_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)
                y1 = np.clip(max(0, y - self.max_h_size // 2), 0, h)
                y2 = np.clip(max(0, y + self.max_h_size // 2), 0, h)
                x1 = np.clip(max(0, x - self.max_w_size // 2), 0, w)
                x2 = np.clip(max(0, x + self.max_w_size // 2), 0, w)
                img[y1: y2, x1: x2, :] = self.fill_value
        return img, box


def cutout(img, holes, fill_value=0):
    # Make a copy of the input image since we don't want to modify it directly
    img = img.copy()
    for x1, y1, x2, y2 in holes:
        img[y1: y2, x1: x2] = fill_value
    return img


class CoarseDropout:
    """
    CoarseDropout of the rectangular regions in the image.
    """
    def __init__(self, max_holes=30, max_height=15, max_width=15,
                 min_holes=15, min_height=9, min_width=9,
                 fill_value=0, p=1):
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes if min_holes is not None else max_holes
        self.min_height = min_height if min_height is not None else max_height
        self.min_width = min_width if min_width is not None else max_width
        self.fill_value = fill_value
        self.prob = p
        assert 0 < self.min_holes <= self.max_holes
        assert 0 < self.min_height <= self.max_height
        assert 0 < self.min_width <= self.max_width

    def get_params_dependent_on_targets(self, img):
        height, width = img.shape[:2]

        holes = []
        for n in range(random.randint(self.min_holes, self.max_holes + 1)):
            hole_height = random.randint(self.min_height, self.max_height + 1)
            hole_width = random.randint(self.min_width, self.max_width + 1)

            y1 = random.randint(0, height - hole_height)
            x1 = random.randint(0, width - hole_width)
            y2 = y1 + hole_height
            x2 = x1 + hole_width
            holes.append((x1, y1, x2, y2))

        return holes

    def __call__(self, image, box=None):
        if random.random() < self.prob:
            holes = self.get_params_dependent_on_targets(image)
            image = cutout(image, holes, self.fill_value)
        return image, box


class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, box=None):
        for t in self.transforms:
            x, box = t(x, box)
        return x, box


class OneOf:
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, x, box=None):
        if random.random() < self.prob:
            t = random.choice(self.transforms)
            t.prob = 1.
            x, box = t(x, box)
        return x, box


class OneOrOther:
    def __init__(self, first, second, prob=0.5):
        self.first = first
        first.prob = 1.
        self.second = second
        second.prob = 1.
        self.prob = prob

    def __call__(self, x, box=None):
        if random.random() < self.prob:
            x, box = self.first(x, box)
        else:
            x, box = self.second(x, box)
        return x, box


if __name__=='__main__':
    import glob
    import cv2
    import matplotlib.pyplot as plt
    imglist = glob.glob('train_images/*.npy')
    labellist = glob.glob('train_labels/*.png')
    transform = DualCompose([
                    # Cutout(num_holes=20, max_h_size=20, max_w_size=20, fill_value=0)
                    RandomRotate90()
                    ])
    img = np.load(imglist[0])
    label = cv2.imread(labellist[0])
    plt.subplot(141)
    plt.imshow(img)
    plt.subplot(142)
    plt.imshow(label)
    img2, label2 = transform(img, label)
    plt.subplot(143)
    plt.imshow(img2)
    plt.subplot(144)
    plt.imshow(label2)
    plt.show()