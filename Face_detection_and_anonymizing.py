import glob
import os

import cv2
import numpy as np
import torch

from models.cspnet import CSPNet_p3p4p5
from utils.keras_weights_loader import load_keras_weights
from utils.utils import format_img, parse_wider_offset


class Driver:
    def __init__(self, i='img_data/', o='img_results/', device='cpu'):
        self.workspace = './images/'
        self.input_root = i
        self.output_root = o
        self.output_path = self.workspace + self.output_root
        self.img_paths = glob.glob(self.workspace + self.input_root + '*.*')
        self.model = CSPNet_p3p4p5(num_scale=2)
        load_keras_weights(self.model, self.workspace + '../net_e382_l0.hdf5')
        self.device = device
        self.model.to(self.device).eval()
        self.imgIdx = 0
        self.current_image_path = ''
        pass

    def get_next_img(self):
        if self.imgIdx >= len(self.img_paths):
            return 'no more imgs'
        self.current_image_path = self.img_paths[self.imgIdx]
        print("processing : " + self.current_image_path)
        self.imgIdx += 1
        return cv2.imread(self.current_image_path)

    def merge_faces_with_bboxes(self, img, bboxes):
        for b in bboxes:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
        return img

    def display_on_colab(self, img):
        from google.colab.patches import cv2_imshow
        cv2_imshow(img)
        pass

    def blur_img(self, img, bboxes):
        for bbox in bboxes:
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[2]), int(bbox[3])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            w = x2 - x1
            h = y2 - y1

            mask_img = np.zeros(img.shape, dtype='uint8')
            cv2.circle(mask_img, (cx, cy), int((w + h) / 2 * 0.5), (255, 255, 255), -1)
            img_all_blurred = cv2.GaussianBlur(img, (31, 31), 11)
            img = np.where(mask_img > 0, img_all_blurred, img)
        return img

    def save_img(self, img):
        (input_path, filename) = os.path.split(self.current_image_path)
        output_file_path = self.output_path + filename
        cv2.imwrite(output_file_path, img)
        print('save file to : %s\n' % output_file_path)
        pass

    def detect_faces(self, img, scale=1, flip=False):
        img_h, img_w = img.shape[:2]
        img_h_new, img_w_new = int(np.ceil(scale * img_h / 16) * 16), int(np.ceil(scale * img_w / 16) * 16)
        scale_h, scale_w = img_h_new / img_h, img_w_new / img_w

        img_s = cv2.resize(img, None, None, fx=scale_w, fy=scale_h, interpolation=cv2.INTER_LINEAR)
        # img_h, img_w = img_s.shape[:2]
        # print frame_number
        input_dim = [img_h_new, img_w_new]

        if flip:
            img_sf = cv2.flip(img_s, 1)
            # x_rcnn = format_img_pad(img_sf, C)
            x_rcnn = format_img(img_sf)
        else:
            # x_rcnn = format_img_pad(img_s, C)
            x_rcnn = format_img(img_s)
        x = torch.from_numpy(x_rcnn).to(self.device)
        x = x.permute(0, 3, 1, 2)
        x_cls, x_reg, x_off = self.model(x)
        # print('x reg shape ', x_reg.shape)
        Y = [x_cls.detach().cpu().numpy(), x_reg.detach().cpu().numpy(), x_off.detach().cpu().numpy()]
        boxes = parse_wider_offset(Y, input_dim, score=0.3, nmsthre=0.4)
        if len(boxes) > 0:
            keep_index = np.where(np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) >= 12)[0]
            boxes = boxes[keep_index, :]
        if len(boxes) > 0:
            if flip:
                boxes[:, [0, 2]] = img_s.shape[1] - boxes[:, [2, 0]]
            boxes[:, 0:4:2] = boxes[:, 0:4:2] / scale_w
            boxes[:, 1:4:2] = boxes[:, 1:4:2] / scale_h
        else:
            boxes = np.empty(shape=[0, 5], dtype=np.float32)
        return boxes


foo = Driver(i='img_data/', o='img_results/', device='cpu')  # Use 'cuda:0' instead of 'cpu' if CUDA is available
s = len(foo.img_paths)
for i in range(s):
    img = foo.get_next_img()
    if isinstance(img, str):
        continue
    bboxes = foo.detect_faces(img)
    img_with_boxes = foo.merge_faces_with_bboxes(img, bboxes)
    # blured_img = foo.blur_img(img, bboxes)
    foo.save_img(img_with_boxes)

foo = Driver(i='test_data/', o='test_results/', device='cpu')  # Use 'cuda:0' instead of 'cpu' if CUDA is available
s = len(foo.img_paths)
for i in range(s):
    img = foo.get_next_img()
    if isinstance(img, str):
        continue
    bboxes = foo.detect_faces(img)
    # img_with_boxes = foo.merge_faces_with_bboxes(img, bboxes)
    blured_img = foo.blur_img(img, bboxes)
    foo.save_img(blured_img)
