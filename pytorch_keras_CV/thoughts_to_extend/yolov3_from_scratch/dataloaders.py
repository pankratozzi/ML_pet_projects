# assume we have annotation file with every line contains label, x, y, x, y and convert to xcycwh (xywh)

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from skimage.transform import resize
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from yolo3_camera import _concat, xcycwh_to_xyxy, xywh_to_xyxy
from model import YOLO
import utils


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def xyxy_to_xcycwh(xyxy):
    """Convert [x1 y1 x2, y2] box format to [x_c y_c w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert not isinstance(xyxy[0], (list, tuple))
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1
        h = xyxy[3] - y1
        x = (xyxy[0] + xyxy[2]) / 2.
        y = (xyxy[1] + xyxy[3]) / 2.
        return [x, y, w, h]
    elif isinstance(xyxy, (np.ndarray, torch.Tensor)):
        wh = xyxy[..., 2:4] - xyxy[..., 0:2]
        xy = (xyxy[..., 0:2] + xyxy[..., 2:4]) / 2.
        return _concat(xy, wh)
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')


def xcycwh_to_xywh(xcycwh):
    """Convert [x_c y_c w h] box format to [x1, y1, w, h] format."""
    if isinstance(xcycwh, (list, tuple)):
        # Single box given as a list of coordinates
        assert not isinstance(xcycwh[0], (list, tuple))
        xc, yc = xcycwh[0], xcycwh[1]
        w = xcycwh[2]
        h = xcycwh[3]
        x1 = xc - w / 2.
        y1 = yc - h / 2.
        return [x1, y1, w, h]
    elif isinstance(xcycwh, (np.ndarray, torch.Tensor)):
        wh = xcycwh[..., 2:4]
        x1y1 = xcycwh[..., 0:2] - wh / 2.
        return _concat(x1y1, wh)
    else:
        raise TypeError('Argument xcycwh must be a list, tuple, or numpy array.')


def get_image_label(image, max_objects=10, labels=None, is_train=True, image_size=416):
    h, w = image.shape[:2]
    dimension_diff = np.abs(h-w)
    pad1, pad2 = dimension_diff // 2, dimension_diff - dimension_diff // 2
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    input_image = np.pad(image, pad, 'constant', constant_values=128) / 255.
    padded_h, padded_w = input_image.shape[:2]
    input_image = resize(input_image, (image_size, image_size), mode='reflect', anti_aliasing=True)
    input_image = np.transpose(input_image, (2, 0, 1))
    input_image = torch.from_numpy(input_image).float()

    if not is_train:
        return input_image

    labels[:, [1, 3]] += pad[1][0]
    labels[:, [2, 4]] += pad[0][0]

    labels[:, 1:] = xyxy_to_xcycwh(labels[:, 1:])

    labels[:, [1, 3]] *= (1 / padded_w)  # * image_size
    labels[:, [2, 4]] *= (1 / padded_h)  # * image_size
    """

    x1 = w * (labels[:, 1] - labels[:, 3] / 2)
    y1 = h * (labels[:, 2] - labels[:, 4] / 2)
    x2 = w * (labels[:, 1] + labels[:, 3] / 2)
    y2 = h * (labels[:, 2] + labels[:, 4] / 2)

    x1 += pad[1][0]
    y1 += pad[0][0]
    x2 += pad[1][0]
    y2 += pad[0][0]

    labels[:, 1] = ((x1 + x2) / 2) / padded_w
    labels[:, 2] = ((y1 + y2) / 2) / padded_h
    labels[:, 3] *= w / padded_w
    labels[:, 4] *= h / padded_h
    """
    filled_labels = np.zeros((max_objects, 5))
    if labels is not None:
        filled_labels[range(labels.shape[0])[:max_objects]] = labels[:max_objects]
    filled_labels = torch.from_numpy(filled_labels)

    return input_image, filled_labels


class YoloDataset(Dataset):
    def __init__(self, root, dataframe, max_objects=20, train=True):
        self.dataframe = dataframe
        self.max_objects = max_objects
        self.train = train
        self.root = root
        self.frames = dataframe['frame'].unique()

    def __len__(self):
        return self.dataframe['frame'].nunique()

    def __getitem__(self, ix):
        frame = self.frames[ix]
        row = self.dataframe.loc[self.dataframe.frame == frame, :]
        image = cv2.imread(self.root + frame)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.train:
            labels = row[['class_id', 'xmin', 'ymin', 'xmax', 'ymax']].values.astype(np.float32)
            return get_image_label(image, self.max_objects, labels, self.train)
        else:
            return get_image_label(image)

    def collate_fn(self, batch):
        images, labels = list(zip(*batch))
        images = [img[None].to(device) for img in images]
        labels = [label[None].to(device) for label in labels]
        images, labels = [torch.cat(i) for i in [images, labels]]
        return images, labels


if __name__ == '__main__':
    df = pd.read_csv('dataset/labels_trainval.csv')
    ds = YoloDataset('dataset/images/', df)
    img, tar = ds[350]
    plt.imshow(img.cpu().detach().numpy().transpose(1, 2, 0))
    coords = tar.cpu().detach().numpy()[:, 1:]
    xyxy = xcycwh_to_xywh(coords)
    xyxy = xyxy[np.any(xyxy != 0, axis=1)] * 416

    ax = plt.gca()

    for i in range(xyxy.shape[0]):
        x, y, xx, yy = xyxy[i]
        ax.add_patch(plt.Rectangle((x, y), xx, yy, fill=False, color='red', linewidth=1))
    plt.axis('off')
    plt.show()

    model = YOLO('yolo.cfg', 80)
    model.load_model('yolov3.weights')
    model.eval()
    with torch.no_grad():
        outputs = model(img.unsqueeze(0), device, targets=None)
        outputs = utils.non_max_suppression(outputs, 0.6, 80, True, 0.4)

    img = img.cpu().detach().numpy().transpose(1, 2, 0)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)

    ax = plt.gca()
    for output in outputs:
        xyxy = xywh_to_xyxy(output[1:5].cpu().detach().numpy().reshape(1, -1))
        xmin, ymin, xmax, ymax = xyxy[0]

        ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, color='red', linewidth=2))

    plt.tight_layout()
    plt.show()
