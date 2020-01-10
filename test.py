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

checkpoint_file = '/home/stephenliang/model.pt'

def read_testset(filename):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)
    df = df.loc[:, ["Image"]]
    df = df.drop_duplicates(["Image"])
    df = df.reset_index(drop = True)
    return df, dc

if __name__ == "__main__":
    df, dc = parse_csv('/home/stephenliang/sample_submission.csv')
    train_dataset = IntracranialDataset(df=df,
                                        base_path='/home/stephenliang/stage_1_train_images/',
                                        training=False)
    data_loader_test  = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=16,
                                                    shuffle=False,
                                                    num_workers=4)

    device = torch.device("cuda:0")

    model = EfficientNet.from_name('efficientnet-b4', {'num_classes': 6})
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model'])

    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    test_pred = np.zeros((len(test_dataset) * 6, 1))

    tta = 6

    for step, batch in enumerate(data_loader_test):
        inputs = batch
        inputs = inputs.to(device, dtype=torch.float)
        
        with torch.no_grad():
            pred = model(inputs)
            test_pred[(step * len(x_batch) * 6):((step + 1) * len(x_batch) * 6)] = torch.sigmoid(pred).detach().cpu().reshape((len(batch) * 6, 1))

    submission =  dc
    submission = pd.concat([submission.drop(columns=['Label']), pd.DataFrame(test_pred)], axis=1)
    submission.columns = ['ID', 'Label']

    submission.to_csv('submission.csv', index=False)
