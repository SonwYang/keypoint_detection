import torch
import numpy as np

class BaseWheatTTA:
    """ author: @shonenkov """
    def __init__(self, image_size):
        self.image_size = image_size

    def augment(self, image):
        raise NotImplementedError

    def batch_augment(self, images):
        raise NotImplementedError

    def deaugment_boxes(self, boxes):
        raise NotImplementedError


class TTANone(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return image

    def batch_augment(self, images):
        return images

    def deaugment_boxes(self, boxes):
        return boxes


class TTAHorizontalFlip(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return image.flip(1)

    def batch_augment(self, images):
        return images.flip(2)

    def deaugment_boxes(self, boxes):
        boxes[:, [1, 3, 5, 7]] = self.image_size - boxes[:, [1, 3, 5, 7]]
        return boxes


class TTAVerticalFlip(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return image.flip(2)

    def batch_augment(self, images):
        return images.flip(3)

    def deaugment_boxes(self, boxes):
        boxes[:, [0, 2, 4, 6]] = self.image_size - boxes[:, [0, 2, 4, 6]]
        return boxes


class TTARotate90(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return torch.rot90(image, 1, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 1, (2, 3))

    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0, 2, 4, 6]] = self.image_size - boxes[:, [1, 3, 5, 7]]
        res_boxes[:, [1, 3, 5, 7]] = boxes[:, [0, 2, 4, 6]]
        return res_boxes


class TTACompose(BaseWheatTTA):
    """ author: @shonenkov """

    def __init__(self, transforms):
        self.transforms = transforms

    def augment(self, image):
        for transform in self.transforms:
            image = transform.augment(image)
        return image

    def batch_augment(self, images):
        for transform in self.transforms:
            images = transform.batch_augment(images)
        return images

    def prepare_boxes(self, boxes):
        result_boxes = boxes.copy()
        result_boxes[:, 0] = np.min(boxes[:, [0, 2]], axis=1)
        result_boxes[:, 2] = np.max(boxes[:, [0, 2]], axis=1)
        result_boxes[:, 1] = np.min(boxes[:, [1, 3]], axis=1)
        result_boxes[:, 3] = np.max(boxes[:, [1, 3]], axis=1)
        return result_boxes

    def deaugment_boxes(self, boxes):
        for transform in self.transforms[::-1]:
            boxes = transform.deaugment_boxes(boxes)
        return self.prepare_boxes(boxes)

