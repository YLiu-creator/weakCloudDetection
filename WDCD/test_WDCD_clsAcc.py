from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from torch.utils import data
from datasets import GF1_WSCD_valid_cls,WDCD_test_cls
import torch
import torch.nn as nn
import time


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    # Datset Options
    parser.add_argument("--dataset", type=str, default='GF1', help='Name of dataset')
    parser.add_argument("--data_root", type=str, default='/home/FAKEDATA/GF1_datasets/datasets_bigpatch_321/data',
                        help="path to Dataset")

    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: None)")
    parser.add_argument("--in_channels", type=int, default=4,
                        help="num input channels (default: None)")
    parser.add_argument("--feature_scale", type=int, default=2,
                        help="feature_scale (default: 2)")

    # Model Options
    parser.add_argument("--model", type=str, default='WDCD_VGG16_GF1',
                        choices=['WDCD_VGG16_GF1','WDCD_VGG16_Landsat','WDCD_VGG16_WDCD'], help='model name')
    # Train Options
    parser.add_argument("--clsACC_only", action='store_true', default=True)
    parser.add_argument("--ckpt", default='./checkpoints/WDCD_vgg16_GF1/best_WDCD_vgg16_GF1_epochs5_acc9503.pth',
                        help="restore from checkpoint")

    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"predict_path\"")
    parser.add_argument("--predict_path", default=None, help="save prediction results")

    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")

    parser.add_argument("--batch_size", type=int, default=64,
                        help='batch size (default: 4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")

    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset=='WDCD':
        val_dst = WDCD_test_cls(root=opts.data_root,image_set='test',input_form='TIFF')
    if opts.dataset=='GF1':
        val_dst = GF1_WSCD_valid_cls(root=opts.data_root,image_set='test',input_form='TIFF')

    return val_dst


def clsAccuracy(opts, model, loader, device):
    """Do validation and return specified samples"""

    right,forgot,alarm,all=0, 0, 0, 0

    with torch.no_grad():
        for i, (images, block_label, mask) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = block_label.to(device, dtype=torch.long)

            img_cls = model(images)

            preds = img_cls.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            for index in range(targets.shape[0]):
                if preds[index]==targets[index]:
                    right+=1
                    all+=1
                elif preds[index]<targets[index]:
                    forgot +=1
                    all += 1
                elif preds[index]>targets[index]:
                    alarm +=1
                    all += 1

    return right, forgot, alarm

def main():

    opts = get_argparser().parse_args()

    # select the GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s,  CUDA_VISIBLE_DEVICES: %s\n" % (device, opts.gpu_id))

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    val_dst = get_dataset(opts)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size, shuffle=False,num_workers=8,drop_last=False,pin_memory=False)
    print("Dataset: %s, Val set: %d" % (opts.dataset, len(val_dst)))

    # Set up model
    model_map = {
        'WDCD_VGG16_GF1': network.GF1_VGG16,
        'WDCD_VGG16_WDCD': network.WDCD_VGG16
    }

    print('Model = %s, num_classes=%d' % (opts.model, opts.num_classes))
    model = model_map[opts.model](num_classes=opts.num_classes)

    # Restore
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint["model_state"].items() if (k in model_dict)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model = nn.DataParallel(model)
        model.to(device)
        print("Model restored from %s" % opts.ckpt)
    else:
        print("Error: Can not load best checkpoints.")


    if opts.clsACC_only:
        model.eval()
        time_before_val = time.time()
        right, forget, alarm =clsAccuracy(opts=opts, model=model, loader=val_loader, device=device)

        accuracy = right/len(val_dst)
        error = forget/len(val_dst)
        false = alarm/len(val_dst)

        time_after_val = time.time()

        print('Time_val = %f' % (time_after_val - time_before_val))
        print("Right:" + str(right))
        print("Forget:" + str(forget))
        print("Alarm:" + str(alarm))
        print("Accuracy:" + str(accuracy))
        print("Fail:" + str(error))
        print("False:" + str(false))


if __name__ == '__main__':
    main()