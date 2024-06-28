import os


import torch
import argparse
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from loss import Loss,cal_dice
from dataset import RAHeart


import numpy as np

from loss import Loss,cal_dice

from model.RASnet import RASnet



def test_loop(model, test_loader, device):
    model.eval()
    running_loss = 0
    pbar = tqdm(test_loader)
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
            # pbar.set_postfix(loss="{:.4f}".format(loss.item()), dice="{:.4f}".format(dice.item()))


    loss = running_loss / len(test_loader)
    dice = dice_valid / len(test_loader)

    return {'loss': loss, 'dice': dice}



def main(args):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    # for kfold in [0,1,2,3,4]:
    db_test = RAHeart(base_dir=args.train_path,
                      split='vaild',
                      transform=transforms.Compose([

                      ]
                      ))

    print('Using {} images for testing.'.format(len(db_test)))

    testloader = DataLoader(db_test, batch_size=1, num_workers=4, pin_memory=True)

    model = RASnet(1, 2).to(device)


    optimizer = optim.Adam(model.parameters(), eps=1e-3)

    weight_path = args.weight_path + 'best.pth'
    if os.path.exists(weight_path):
        weight_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(weight_dict)
        print('Successfully loading checkpoint.')
        test_metrics = test_loop(model, testloader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patch_size', type=float, default=(320, 320, 96))
    parser.add_argument('--train_path', type=str, default='')


    args = parser.parse_args()

    main(args)