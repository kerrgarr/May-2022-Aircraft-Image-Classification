
from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from .utils import AircraftDataset, FAMILY_NAMES
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import os, subprocess, sys
import matplotlib.pyplot as plt

def train(args):
    from os import path
    model = model_factory[args.model]()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    if args.continue_training:
        from os import path
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % args.model)))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    loss = torch.nn.CrossEntropyLoss() # call loss function

    train_data = load_data('data/train')
    valid_data = load_data('data/valid')
    
    global_step = 0
    
    for epoch in range(args.num_epoch):
        model.train()
        loss_vals, acc_vals, vacc_vals = [], [], []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            logit = model(img)
            loss_val = loss(logit, label)
            acc_val = accuracy(logit, label)

            loss_vals.append(loss_val.detach().cpu().numpy())
            acc_vals.append(acc_val.detach().cpu().numpy())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
      
        avg_loss = sum(loss_vals) / len(loss_vals)
        train_logger.add_scalar('loss', avg_loss, global_step=global_step)
        avg_acc = sum(acc_vals) / len(acc_vals)
        train_logger.add_scalar('accuracy', avg_acc, global_step=global_step)

        model.eval()
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)
            val_prediction = model(img).max(1)[1].type_as(label)
            val_true = label
            #print(val_prediction)
            #print(label)
            vacc_vals.append( accuracy(model(img), label).detach().cpu().numpy() )
        avg_vacc = sum(vacc_vals) / len(vacc_vals)
        valid_logger.add_scalar('accuracy', avg_vacc, global_step=global_step)

        print('epoch %-3d \t loss = %0.3f \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_loss, avg_acc, avg_vacc))
        
    save_model(model)

    arr = confusion_matrix(val_true.view(-1).detach().cpu().numpy(), val_prediction.view(-1).detach().cpu().numpy())
    class_names = FAMILY_NAMES
    df_cm = pd.DataFrame(arr, class_names, class_names)
    plt.figure()
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.show() 


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('-m', '--model', choices=['cnn', 'resnet', 'vgg'], default='cnn')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)
