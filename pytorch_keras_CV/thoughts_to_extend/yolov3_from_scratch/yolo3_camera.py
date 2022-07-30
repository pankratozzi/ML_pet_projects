import torch
import torch.nn as nn
import numpy as np
import cv2
import time
from utils import prepare_image, non_max_suppression
import matplotlib.pyplot as plt
from collections import Counter
from model import YOLO


device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 80
image_size = 416
class_names = [line.rstrip() for line in open('coco.names', 'r')]
class_dict = dict(zip(np.arange(num_classes), class_names))
model = YOLO('yolo.cfg', num_classes)
model.load_model('yolov3.weights')
model.eval()


def _concat(x, y):
    """ Concat by the last dimension """
    if isinstance(x, np.ndarray):
        return np.concatenate((x, y), axis=-1)
    elif isinstance(x, torch.Tensor):
        return torch.cat([x, y], dim=-1)
    else:
        raise TypeError("unknown type '{}'".format(type(x)))


def xcycwh_to_xyxy(xcycwh):
    """Convert [x_c y_c w h] box format to [x1, y1, x2, y2] format."""
    if isinstance(xcycwh, (list, tuple)):
        # Single box given as a list of coordinates
        assert not isinstance(xcycwh[0], (list, tuple))
        xc, yc = xcycwh[0], xcycwh[1]
        w = xcycwh[2]
        h = xcycwh[3]
        x1 = xc - w / 2.
        y1 = yc - h / 2.
        x2 = xc + w / 2.
        y2 = yc + h / 2.
        return [x1, y1, x2, y2]
    elif isinstance(xcycwh, (np.ndarray, torch.Tensor)):
        wh = xcycwh[..., 2:4]
        x1y1 = xcycwh[..., 0:2] - wh / 2.
        x2y2 = xcycwh[..., 0:2] + wh / 2.
        return _concat(x1y1, x2y2)
    else:
        raise TypeError('Argument xcycwh must be a list, tuple, or numpy array.')


def xywh_to_xyxy(xywh):
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format."""
    if isinstance(xywh, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xywh) == 4
        x1, y1 = xywh[0], xywh[1]
        x2 = x1 + np.maximum(0., xywh[2] - 1.)
        y2 = y1 + np.maximum(0., xywh[3] - 1.)
        return (x1, y1, x2, y2)
    elif isinstance(xywh, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack(
            (xywh[:, 0:2], xywh[:, 0:2] + np.maximum(0, xywh[:, 2:4] - 1))
        )
    else:
        raise TypeError('Argument xywh must be a list, tuple, or numpy array.')


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cmap = plt.get_cmap('tab20b')

    assert cap.isOpened(), "Camera is not available."
    frames = 0
    start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            original_image, input_image = prepare_image(frame, image_size)
            outputs = None

            with torch.no_grad():
                outputs = model(input_image, device)
                outputs = non_max_suppression(outputs, 0.9, num_classes, True, 0.05)

            count = Counter()
            output_text = ""

            if isinstance(outputs, int):
                frames += 1
                print(f'FPS of the video is {(frames / (time.time() - start)):5.2f}')
                cv2.imshow("frame", original_image)

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            h, w = original_image.shape[:2]
            for output in outputs:
                count[class_dict[int(output[-1])]] += 1

                xyxy = xywh_to_xyxy(output[1:5].cpu().detach().numpy().tolist())
                xmin, ymin, xmax, ymax = xyxy
                xmin, xmax = [int(c * (w / image_size)) for c in [xmin, xmax]]
                ymin, ymax = [int(c * (h / image_size)) for c in [ymin, ymax]]

                cls = int(output[-1])
                label = class_dict[cls]
                cv2.rectangle(original_image, (xmin, ymin), (xmax+xmin, ymax+ymin), [0, 0, 255], 1)
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                c2 = xmin + t_size[0] + 3, ymin + t_size[1] + 4
                cv2.rectangle(original_image, (xmin, ymin), c2, [0, 0, 255], -1)
                cv2.putText(original_image, label, (xmin, ymin + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottom_left_corner_text = (10, 30)
            font_scale = .5
            font_color = (0, 0, 255)
            line_type = 1

            for k, v in count.items():
                line = f'{k}: {v}'
                output_text += line
                output_text += '\n'

            for i, line in enumerate(output_text.split('\n')):
                y = 15 + i * 15
                cv2.putText(original_image, line, (10, y), font, font_scale, font_color, line_type)

            cv2.imshow("frame", original_image)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print(f'FPS of the video is {(frames / (time.time() - start)):5.2f}')
