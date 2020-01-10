import os
import cv2
import glob
import pydicom
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from generator import IntracranialDataset
from apex import amp
from efficientnet_pytorch import EfficientNet

n_epochs = 2
checkpoint_file = '/home/stephenliang/model.pt'

def parse_csv(filename):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)
    duplicates_to_remove = [
        1598538, 1598539, 1598540, 1598541, 1598542, 1598543,
        312468,  312469,  312470,  312471,  312472,  312473,
        2708700, 2708701, 2708702, 2708703, 2708704, 2708705,
        3032994, 3032995, 3032996, 3032997, 3032998, 3032999
    ]
    df = df.drop(index = duplicates_to_remove)
    df = df.reset_index(drop = True)    
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    epidural_df = df[df.Label['epidural'] == 1]
    df = pd.concat([epidural_df, epidural_df, epidural_df, df])
    return df.sample(frac=1)


if __name__ == "__main__":
    df = parse_csv('/home/stephenliang/stage_1_train.csv')
    train_dataset = IntracranialDataset(df=df,
                                        base_path='/home/stephenliang/stage_1_train_images/',
                                        training=True)
    data_loader_train = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=16,
                                                    shuffle=True,
                                                    num_workers=8)

    device = torch.device("cuda:0")
    model = EfficientNet.from_name('efficientnet-b4', {'num_classes': 6})
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    plist = [{'params': model.parameters(), 'lr': 3e-4}]
    optimizer = optim.Adamax(plist, lr=3e-4)

    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch + 1, n_epochs))
        print('-\n')

        tr_loss = 0
        
        for step, batch in enumerate(data_loader_train):
            inputs, labels = batch

            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            tr_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()
            
            if step & 127 == 127:
                epoch_loss = tr_loss / 128
                print('epoch: {} : step: {} - loss:{:.4f}'.format(epoch + 1, step + 1, epoch_loss))
                tr_loss = 0.0

            if (step & 2047 == 2047): # save this every 2048 steps
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict() }, checkpoint_file)

