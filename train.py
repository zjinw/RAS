import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from loss import Loss,cal_dice
import torch.optim as optim
from dataset import RAHeart
from collections import Counter
import numpy as np
from model.RASnet import RASnet



def train_loop(model, optimizer, criterion, train_loader, device):
    model.train()
    running_loss = 0
    pbar = tqdm(train_loader)
    dice_train = 0

    optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True,
                                               threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                               eps=1e-06)

    for sampled_batch in pbar:
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        # print(volume_batch.shape,label_batch.shape)
        outputs = model(volume_batch)
        
        # 
        # outputs1 = outputs[1]
        # 
        # 
        # loss1 = criterion(outputs1, label_batch)
 

        loss = criterion(outputs, label_batch)
        dice = cal_dice(outputs, label_batch)
        dice_train += dice.item()
        # pbar.set_postfix(loss="{:.4f}".format(loss.item()), dice="{:.4f}".format(dice.item()))

        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss = running_loss / len(train_loader)
    dice = dice_train / len(train_loader)
    return {'loss': loss, 'dice': dice}


def eval_loop(model, criterion, valid_loader, device):
    model.eval()
    running_loss = 0
    pbar = tqdm(valid_loader)
    dice_valid = 0

    with torch.no_grad():
        for sampled_batch in pbar:
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)

            outputs = model(volume_batch)

            loss = criterion(outputs, label_batch)
            dice = cal_dice(outputs, label_batch)
            running_loss += loss.item()
            dice_valid += dice.item()
            pbar.set_postfix(loss="{:.4f}".format(loss.item()), dice="{:.4f}".format(dice.item()))

    loss = running_loss / len(valid_loader)
    dice = dice_valid / len(valid_loader)
    return {'loss': loss, 'dice': dice}


def train(args, model, optimizer, criterion, train_loader, valid_loader, epochs,
          device, loss_min=999.0):


    for e in range(epochs):
        # train for epoch
        train_metrics = train_loop(model, optimizer, criterion, train_loader, device)
        valid_metrics = eval_loop(model, criterion, valid_loader, device)


        if valid_metrics['loss'] < loss_min:
            loss_min = valid_metrics['loss']
            torch.save(model.state_dict(),args.weight_path + 'best.pth')

    print("Finished Training!")


def main(args):
    

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # for kfold in [0,1,2,3,4]:
        # data info
    db_train = RAHeart(base_dir=args.train_path,
                           split='train',
                           transform=transforms.Compose([
                           #  data process
                           ]))
    db_test = RAHeart(base_dir=args.train_path,
                          split='vaild',
                          transform=transforms.Compose([
                            #  data process
                          ]))
    print('Using {} images for training, {} images for testing.'.format(len(db_train), len(db_test)))
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                             drop_last=True)
    testloader = DataLoader(db_test, batch_size=1, num_workers=4, pin_memory=True)

    model = RASnet(1, 2).to(device)

    criterion = Loss(n_classes=args.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), eps=1e-3,weight_decay=1e-4)

    train(args, model, optimizer, criterion, trainloader, testloader, args.epochs, device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patch_size', type=float)
    parser.add_argument('--train_path', type=str, default='')

    args = parser.parse_args()

    main(args)