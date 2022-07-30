import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import os
from skimage.transform import resize
from model import EmptyLayer, YoloLayer


def parse_config(config_path):
    assert os.path.exists(config_path), f'{config_path} does not exist.'
    with open(config_path, 'r') as infile:
        lines = infile.read().split('\n')
    lines = [x.rstrip().lstrip() for x in lines if x and not x.startswith('#')]
    blocks = []
    for line in lines:
        if line.startswith('['):
            blocks.append({'type': line[1: -1].rstrip()})
        else:
            key, value = line.split('=')
            blocks[-1][key.rstrip()] = value.lstrip()
    return blocks


def construct_module_list(blocks):
    network_params = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks, -1):
        module = nn.Sequential()
        if x['type'] == 'net':
            continue
        if x['type'] == 'convolutional':
            activation = x['activation']
            try:
                batch_normalization = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalization = 0
                bias = True
            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            pad = (kernel_size - 1) // 2 if padding else 0
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module(f'conv_{index}', conv)
            if batch_normalization:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f'batch_norm_{index}', bn)
            if activation == 'leaky':
                activation_fn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f'leaky_{index}', activation_fn)
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=stride, mode='nearest')
            module.add_module(f'upsample_{index}', upsample)
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            start = int(x['layers'][0])
            try:
                end = int(x['layers'][1])
            except:
                end = 0
            if start > 0:
                start -= index
            if end > 0:
                end -= index
            route = EmptyLayer()
            module.add_module(f'route_{index}', route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module(f'shortcut_{index}', shortcut)

        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x['anchors'].split(',')
            anchors = [int(anc) for anc in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = YoloLayer(anchors)
            module.add_module(f'Detection_{index}', detection)

        else:
            raise ValueError(f'Unknown type: {x["type"]}')

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return network_params, module_list


def unique(tensor):
    tensor_numpy = tensor.cpu().detach().numpy()
    unique_numpy = np.unique(tensor_numpy)
    unique_tensor = torch.from_numpy(unique_numpy)

    tensor_result = tensor.new(unique_tensor.shape)
    tensor_result.copy_(unique_tensor)
    return tensor_result


def non_max_suppression(prediction, confidence, num_classes, nms=True, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    try:
        _ = torch.nonzero(prediction[:, :, 4]).transpose(0, 1).contiguous()
    except:
        return 0

    box_a = prediction.new(prediction.shape)
    box_a[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_a[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_a[:, :, 2] = prediction[:, :, 2] - prediction[:, :, 2] / 2
    box_a[:, :, 3] = prediction[:, :, 3] - prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_a[:, :, :4]

    batch_size = prediction.size(0)
    output = prediction.new(1, prediction.size(2) + 1)
    write = False

    for idx in range(batch_size):
        image_prediction = prediction[idx]
        max_conf, max_conf_score = torch.max(image_prediction[:, 5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)

        sequence = image_prediction[:, :5], max_conf, max_conf_score
        image_prediction = torch.cat(sequence, 1)

        non_zero_index = torch.nonzero(image_prediction[:, 4])
        image_prediction_ = image_prediction[non_zero_index.squeeze(), :].view(-1, 7)

        try:
            image_classes = unique(image_prediction_[:, -1])
        except:
            continue

        for cls in image_classes:
            class_mask = image_prediction_ * (image_prediction_[:, -1] == cls).float().unsqueeze(1)
            class_mask_index = torch.nonzero(class_mask[:, -2]).squeeze()

            image_predicted_class = image_prediction_[class_mask_index].view(-1, 7)
            conf_sorted_index = torch.sort(image_predicted_class[:, 4], descending=True)[1]
            image_predicted_class = image_predicted_class[conf_sorted_index]
            index = image_predicted_class.size(0)

            if nms:
                for i in range(index):
                    try:
                        ious = bboxIOU(image_predicted_class[i].unsqueeze(0), image_predicted_class[i+1:], True)
                    except ValueError:
                        break

                    except IndexError:
                        break

                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_predicted_class[i+1:] *= iou_mask

                    non_zero_index = torch.nonzero(image_predicted_class[:, 4]).squeeze()
                    image_predicted_class = image_predicted_class[non_zero_index].view(-1, 7)

            batch_index = image_predicted_class.new(image_predicted_class.size(0), 1).fill_(idx)
            sequence = batch_index, image_predicted_class
            if not write:
                output = torch.cat(sequence, 1)
                write = True
            else:
                out = torch.cat(sequence, 1)
                output = torch.cat((output, out))

    return output


def build_targets(predicted_boxes, predicted_conf, predicted_cls, targets, anchors, num_anchors, num_classes, grid_size, ignore_threshold):
    batch_size = targets.size(0)
    mask = torch.zeros(batch_size, num_anchors, grid_size, grid_size)
    conf_mask = torch.ones(batch_size, num_anchors, grid_size, grid_size)
    tx = torch.zeros(batch_size, num_anchors, grid_size, grid_size)
    ty = torch.zeros(batch_size, num_anchors, grid_size, grid_size)
    tw = torch.zeros(batch_size, num_anchors, grid_size, grid_size)
    th = torch.zeros(batch_size, num_anchors, grid_size, grid_size)
    tconf = torch.ByteTensor(batch_size, num_anchors, grid_size, grid_size).fill_(0)
    tcls = torch.ByteTensor(batch_size, num_anchors, grid_size, grid_size, num_classes).fill_(0)

    num_ground_truth = 0
    num_correct = 0
    for batch_idx in range(batch_size):
        for target_idx in range(targets.shape[1]):
            if targets[batch_idx, target_idx].sum() == 0:
                continue
            num_ground_truth += 1

            gx = targets[batch_idx, target_idx, 1] * grid_size
            gy = targets[batch_idx, target_idx, 2] * grid_size
            gw = targets[batch_idx, target_idx, 3] * grid_size
            gh = targets[batch_idx, target_idx, 4] * grid_size

            grid_x_index = int(gx)
            grid_y_index = int(gy)

            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            anchor_iou = bboxIOU(gt_box, anchor_shapes, True)

            conf_mask[batch_idx, anchor_iou > ignore_threshold, grid_y_index, grid_x_index] = 0
            best = np.argmax(anchor_iou)

            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            predicted_box = predicted_boxes[batch_idx, best, grid_y_index, grid_x_index].type(torch.FloatTensor).unsqueeze(0)
            mask[batch_idx, best, grid_y_index, grid_x_index] = 1
            conf_mask[batch_idx, best, grid_y_index, grid_x_index] = 1

            tx[batch_idx, best, grid_y_index, grid_x_index] = gx - grid_x_index
            ty[batch_idx, best, grid_y_index, grid_x_index] = gy - grid_y_index
            tw[batch_idx, best, grid_y_index, grid_x_index] = np.log(gw / anchors[best][0] + 1e-15)
            th[batch_idx, best, grid_y_index, grid_x_index] = np.log(gh / anchors[best][1] + 1e-15)

            target_label = int(targets[batch_idx, target_idx, 0])
            tcls[batch_idx, best, grid_y_index, grid_x_index, target_label] = 1
            tconf[batch_idx, best, grid_y_index, grid_x_index] = 1

            iou = bboxIOU(gt_box, predicted_box, False)
            predicted_label = torch.argmax(predicted_cls[batch_idx, best, grid_y_index, grid_x_index])
            score = predicted_conf[batch_idx, best, grid_y_index, grid_x_index]

            if iou > 0.5 and predicted_label == target_label and score >= 0.5:
                num_correct += 1

    return num_ground_truth, num_correct, mask, conf_mask, tx, ty, tw, th, tconf, tcls


def bboxIOU(box1, box2, xyxy):
    if xyxy:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    intersection_x1 = torch.max(b1_x1, b2_x1)
    intersection_y1 = torch.max(b1_y1, b2_y1)
    intersection_x2 = torch.min(b1_x2, b2_x2)
    intersection_y2 = torch.min(b1_y2, b2_y2)

    intersection_area = (intersection_x2 - intersection_x1 + 1) * (intersection_y2 - intersection_y1 + 1)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = intersection_area / (b1_area + b2_area - intersection_area + 1e-15)

    return iou


def prepare_image(image, image_size):
    original_image = image
    h, w = image.shape[:2]

    dimension_diff = np.abs(h - w)
    pad1, pad2 = dimension_diff // 2, dimension_diff - dimension_diff // 2
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    input_image = np.pad(image, pad, mode='constant', constant_values=128) / 255.
    input_image = resize(input_image, (image_size, image_size, 3), mode='reflect', anti_aliasing=True)
    # input_image = T.ToTensor()(input_image).unsqueeze(0)  # transpose and normalize if not already
    input_image = np.transpose(input_image, (2, 0, 1))
    input_image = torch.from_numpy(input_image).float().unsqueeze(0)

    return original_image, input_image


if __name__ == '__main__':
    blocks = parse_config('yolo.cfg')
    module = construct_module_list(blocks)
    print(module)
