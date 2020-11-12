import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_results(result, ori_image, cat):
    ori_image_ = ori_image.copy()
    for pred in result:
        score = pred[-1]
        tl = np.asarray([pred[0], pred[1]], np.float32)
        tr = np.asarray([pred[2], pred[3]], np.float32)
        br = np.asarray([pred[4], pred[5]], np.float32)
        bl = np.asarray([pred[6], pred[7]], np.float32)

        tt = (np.asarray(tl, np.float32) + np.asarray(tr, np.float32)) / 2
        rr = (np.asarray(tr, np.float32) + np.asarray(br, np.float32)) / 2
        bb = (np.asarray(bl, np.float32) + np.asarray(br, np.float32)) / 2
        ll = (np.asarray(tl, np.float32) + np.asarray(bl, np.float32)) / 2

        box = np.asarray([tl, tr, br, bl], np.float32)
        cen_pts = np.mean(box, axis=0)
        cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(tt[0]), int(tt[1])), (0, 0, 255),
                 1, 1)
        cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(rr[0]), int(rr[1])), (255, 0, 255),
                 1, 1)
        cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(bb[0]), int(bb[1])), (0, 255, 0),
                 1, 1)
        cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(ll[0]), int(ll[1])), (255, 0, 0),
                 1, 1)

        ori_image = cv2.drawContours(ori_image, [np.int0(box)], -1, (255, 0, 255), 2, 1)
        cv2.putText(ori_image, '{:.2f} {}'.format(score, cat), (box[1][0], box[1][1]),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1, 1)
    plt.subplot(121)
    plt.imshow(ori_image_)
    plt.subplot(122)
    plt.imshow(ori_image)
    plt.show()


def show_results2(bboxes, scores, ori_image):
    ori_image_ = ori_image.copy()
    for pred, score in zip(bboxes, scores):
        cat = pred[-1]
        tl = np.asarray([pred[0], pred[1]], np.float32)
        tr = np.asarray([pred[2], pred[3]], np.float32)
        br = np.asarray([pred[4], pred[5]], np.float32)
        bl = np.asarray([pred[6], pred[7]], np.float32)

        tt = (np.asarray(tl, np.float32) + np.asarray(tr, np.float32)) / 2
        rr = (np.asarray(tr, np.float32) + np.asarray(br, np.float32)) / 2
        bb = (np.asarray(bl, np.float32) + np.asarray(br, np.float32)) / 2
        ll = (np.asarray(tl, np.float32) + np.asarray(bl, np.float32)) / 2

        box = np.asarray([tl, tr, br, bl], np.float32)
        cen_pts = np.mean(box, axis=0)
        cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(tt[0]), int(tt[1])), (0, 0, 255),
                 1, 1)
        cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(rr[0]), int(rr[1])), (255, 0, 255),
                 1, 1)
        cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(bb[0]), int(bb[1])), (0, 255, 0),
                 1, 1)
        cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(ll[0]), int(ll[1])), (255, 0, 0),
                 1, 1)

        ori_image = cv2.drawContours(ori_image, [np.int0(box)], -1, (255, 0, 255), 2, 1)
        cv2.putText(ori_image, '{:.2f} {}'.format(score, cat), (box[1][0], box[1][1]),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1, 1)
    plt.subplot(121)
    plt.imshow(ori_image_)
    plt.subplot(122)
    plt.imshow(ori_image)
    plt.show()