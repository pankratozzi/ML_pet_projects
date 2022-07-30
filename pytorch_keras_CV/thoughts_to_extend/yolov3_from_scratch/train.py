import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
from dataloaders import YoloDataset
from model import YOLO


device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 2
image_size = 416
epochs = 30
num_classes = 6


if __name__ == '__main__':
    df = pd.read_csv('dataset/labels_trainval.csv')
    dataset = YoloDataset('dataset/images/', df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn, drop_last=True)

    model = YOLO('cars.cfg', num_classes).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau()
    min_loss = np.inf

    epoch_loss = []
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1} / {epochs}')
        for idx, (images, labels) in tqdm(enumerate(dataloader), leave=False):
            optimizer.zero_grad()
            loss = model(images, device, labels)
            loss.backward()
            optimizer.step()

            model.seed += images.size(0)
            epoch_loss.append(loss.item())

            if idx % 100 == 0:
                print(
                    'Epoch:{}, Batch:{}, x_loss:{:0.4f}, y_loss:{:.4f}, w_loss:{:.4f}, '
                    'h_loss:{:.4f}, conf:{:.4f},cls:{:.4f}, '
                    'precision:{:.4f},recall:{:.4f}, total:{:.4f}'.format(epoch, idx,
                                                                          model.losses["x"], model.losses["y"],
                                                                          model.losses["w"], model.losses["h"],
                                                                          model.losses["conf"], model.losses["cls"],
                                                                          model.losses["recall"], model.losses["precision"],
                                                                          loss.item()))
        epoch_end_loss = np.mean(epoch_loss)
        if epoch_end_loss.item() < min_loss:
            model.save_model('train.weights')
