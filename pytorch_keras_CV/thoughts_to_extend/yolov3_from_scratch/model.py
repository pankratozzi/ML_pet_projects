import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import utils


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class YoloLayer(nn.Module):
    def __init__(self, anchors):
        super(YoloLayer, self).__init__()
        self.anchors = anchors
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.xe_loss = nn.CrossEntropyLoss()
        self.lambda_coord = 5.
        self.lambda_noobj = .5

    def forward(self, prediction, image_size, num_classes, device, targets=None):
        prediction = prediction.to(device)
        anchors = self.anchors
        batch_size = prediction.size(0)
        grid_size = prediction.size(2)
        stride = image_size // grid_size
        bbox_attrs = 5 + num_classes
        num_anchors = len(anchors)

        prediction = prediction.view(batch_size, num_anchors, bbox_attrs, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()
        scaled_anchors = torch.FloatTensor([(a[0] / stride, a[1] / stride) for a in anchors])

        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        predicted_conf = torch.sigmoid(prediction[..., 4])
        predicted_cls = torch.sigmoid(prediction[..., 5:])

        scaled_anchors = scaled_anchors.to(device)
        grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).type(torch.float32).to(device)
        grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).type(torch.float32).to(device)

        anchor_w = scaled_anchors[:, 0:1].view((1, num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, num_anchors, 1, 1))
        predicted_boxes = torch.Tensor(prediction[..., :4].shape).float().to(device)
        predicted_boxes[..., 0] = x.data + grid_x
        predicted_boxes[..., 1] = y.data + grid_y
        predicted_boxes[..., 2] = torch.exp(w.data) * anchor_w
        predicted_boxes[..., 3] = torch.exp(h.data) * anchor_h

        if targets is not None:

            num_gt, num_correct, mask, conf_mask, tx, ty, tw, th, tconf, tcls = utils.build_targets(
                predicted_boxes=predicted_boxes.cpu().data,
                predicted_conf=predicted_conf.cpu().data,
                predicted_cls=predicted_cls.cpu().data,
                targets=targets.cpu().data,
                anchors=scaled_anchors.cpu().data,
                num_anchors=num_anchors,
                num_classes=num_classes,
                grid_size=grid_size,
                ignore_threshold=0.5,
            )

            num_proposals = int((predicted_conf > 0.5).sum().item())
            recall = float(num_correct / num_gt) if num_gt else 1
            precision = float(num_correct / num_proposals)

            mask = mask.long()
            conf_mask = conf_mask.long()

            tx = torch.as_tensor(tx, dtype=torch.float32).requires_grad_(False).to(device)  # really False???
            ty = torch.as_tensor(ty, dtype=torch.float32).requires_grad_(False).to(device)
            tw = torch.as_tensor(tw, dtype=torch.float32).requires_grad_(False).to(device)
            th = torch.as_tensor(th, dtype=torch.float32).requires_grad_(False).to(device)
            tconf = torch.as_tensor(tconf, dtype=torch.float32).requires_grad_(False).to(device)
            tcls = torch.as_tensor(tcls, dtype=torch.float32).requires_grad_(False).to(device)

            conf_mask_true = mask
            conf_mask_false = conf_mask - mask

            loss_x = self.lambda_coord * self.mse_loss(x[mask], tx[mask])
            loss_y = self.lambda_coord * self.mse_loss(y[mask], ty[mask])
            loss_w = self.lambda_coord * self.mse_loss(w[mask], tw[mask])
            loss_h = self.lambda_coord * self.mse_loss(h[mask], th[mask])

            loss_conf = self.lambda_noobj * self.mse_loss(predicted_conf[conf_mask_false], tconf[conf_mask_false]) + \
                self.mse_loss(predicted_conf[conf_mask_true], tconf[conf_mask_true])

            loss_cls = (1 / batch_size) * self.xe_loss(predicted_cls[mask], torch.argmax(tcls[mask], 1))
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item(), recall, precision

        else:
            prediction = torch.cat((predicted_boxes.view(batch_size, -1, 4) * stride,
                                    predicted_conf.view(batch_size, -1, 1),
                                    predicted_cls.view(batch_size, -1, num_classes)), -1)
            prediction = prediction.view(batch_size, num_anchors, grid_size, grid_size, bbox_attrs).permute(0, 2, 3, 1, 4)\
                .contiguous().view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

            return prediction


class YOLO(nn.Module):
    def __init__(self, cfg_file, num_classes):
        super(YOLO, self).__init__()
        self.blocks = utils.parse_config(cfg_file)
        self.num_classes = num_classes
        self.network_info, self.module_list = utils.construct_module_list(self.blocks)
        self.seen = 0
        self.header = torch.IntTensor([0, 0, 0, self.seen, 0])
        self.loss_names = ["x", "y", "w", "h", "confidence", "classification", "recall", "precision"]
        self.losses = None

    def forward(self, x, device, targets=None):
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        blocks = self.blocks[1:]

        for i, (block, module) in enumerate(zip(blocks, self.module_list)):
            if block["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)

            elif block["type"] == "route":
                layer_i = [int(x) for x in block["layers"]]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)

            elif block["type"] == "shortcut":
                layer_i = int(block["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]

            elif block["type"] == "yolo":
                input_dim = int(self.network_info["height"])
                if targets is not None:
                    x, *losses = module[0](x, input_dim, self.num_classes, device, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                else:
                    x = module[0](x, input_dim, self.num_classes, device, targets=None)
                output.append(x)
            layer_outputs.append(x)

        self.losses["recall"] /= 3
        self.losses["precision"] /= 3
        return sum(output) if targets is not None else torch.cat(output, 1)

    def load_model(self, weightfile):
        fp = open(weightfile, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = model[0]

                if batch_normalize:
                    bn = model[1]

                    num_bn_biases = bn.bias.numel()

                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    num_biases = conv.bias.numel()

                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    conv_biases = conv_biases.view_as(conv.bias.data)

                    conv.bias.data.copy_(conv_biases)

                num_weights = conv.weight.numel()

                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

    def save_model(self, savedfile):
        fp = open(savedfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header = header.numpy()
        header.tofile(fp)

        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"]
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = model[0]
                if batch_normalize:
                    bn = model[1]
                    bn.bias.data.cpu().numpy().tofile(fp)
                    bn.weight.data.cpu().numpy().tofile(fp)
                    bn.running_mean.cpu().numpy().tofile(fp)
                    bn.running_var.cpu().numpy().tofile(fp)
                else:
                    conv.bias.data.cpu().numpy().tofile(fp)
                conv.weight.data.cpu().numpy().tofile(fp)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolo.cfg', 80)
    model.load_model('yolov3.weights')
    model.eval()
    sample = torch.randn((3, 416, 416)).unsqueeze(0)
    with torch.no_grad():
        sample_output = model(sample, device, targets=None)
        print(sample_output.shape, sample_output)
