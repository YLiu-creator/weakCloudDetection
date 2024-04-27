from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from torch.utils import data
from datasets import WSFNet_cls_GF1
from metrics import StreamSegMetrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import matplotlib.pyplot as plt

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--dataset", type=str, default='gf1', help='Name of dataset')
    parser.add_argument("--data_root", type=str, default='/GF1_datasets/datasets_321/data',
                        help="path to Dataset")


    parser.add_argument("--input_form", type=str, default='TIFF', choices=['RGB', 'TIFF'],
                        help="input form of GF-1 WFV")

    parser.add_argument("--num_classes", type=int, default=2,help="num classes (default: None)")

    # Model Options
    parser.add_argument("--model", type=str, default='WSFNet_stage1_GF1',
                        choices=['WDCD_vgg16_cam','scExtractor','WSCD_resnet50','WSCD_resnet50_cam'], help='model name')

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=True)
    parser.add_argument("--ckpt", default='./checkpoints/WSFNet_stage1_GF1/best_WSFNet_stage1_GF1_epochs12.pth',
                        help="restore from checkpoint")

    parser.add_argument("--save_CAM", action='store_true', default=True,
                        help="save segmentation results to \"save_path\"")
    parser.add_argument("--save_path", default='/home/liuyang/weakly_spuervisied_CD/WSFNet/',
                        help="save prediction results")

    parser.add_argument("--gpu_id", type=str, default='0,1,2,3',
                        help="GPU ID")

    parser.add_argument("--batch_size", type=int, default=128,
                        help='batch size (default: 4)')
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=2,
                        help="epoch interval for eval (default: 5000)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    """ Dataset And Augmentation
        """
    if opts.dataset == 'gf1':
        val_dst = WSFNet_cls_GF1(root=opts.data_root, image_set='test_vis')

    return val_dst


def getCAM(opts, model, loader, device):
    """Do validation and return specified samples"""

    right,forgot,alarm,all = 0, 0, 0, 0
    index = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            img_cls, cam = model(images)

            preds = img_cls.detach().max(dim=1)[1].cpu().numpy()
            labels = labels.cpu().numpy()

            for ii in range(labels.shape[0]):
                if preds[ii] == labels[ii]:
                    right += 1
                    all += 1
                elif preds[ii] < labels[ii]:
                    forgot += 1
                    all += 1
                elif preds[ii] > labels[ii]:
                    alarm += 1
                    all += 1

            if opts.save_CAM:
                sample_fname = loader.sampler.data_source.images[:]
                os.makedirs(opts.save_path, exist_ok=True)
                os.makedirs(opts.save_path + 'CAM_testVIS/', exist_ok=True)

                b, c, h, w = images.size()[0], images.size()[1], images.size()[2], images.size()[3]
                cam_resize = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
                cam_sum = cam_resize.detach().max(dim=1)[1].cpu().numpy()

                for batch in range(images.shape[0]):
                    content = sample_fname[index].replace('.npy', "").split("/")
                    name = content[-1]
                    np.save(opts.save_path + 'CAM_testVIS/' + name + '_1cam.npy', cam_sum[batch, :, :])
                    index = index + 1

        accuracy = right / all
        error = forgot / all
        false = alarm / all

    return accuracy, error, false


if __name__ == '__main__':

    opts = get_argparser().parse_args()

    # select the GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s,  CUDA_VISIBLE_DEVICES: %s\n" % (device, opts.gpu_id))

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # train_dst, val_dst = get_dataset(opts)
    val_dst = get_dataset(opts)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size, shuffle=False, num_workers=64, drop_last=False,
                                 pin_memory=False)

    print("Dataset: %s, val set: %d" % (opts.dataset, len(val_dst)))

    # Set up model
    model_map = {
        'WSFNet_stage1_GF1': network.EDResNet
    }

    print('Model = %s, num_classes=%d' % (opts.model, opts.num_classes))
    model = model_map[opts.model](num_classes=opts.num_classes)

    # Restore
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Model restored from %s" % opts.ckpt)
    else:
        print("Error: Can not load best checkpoints.")

    print("validation...")
    model.eval()

    time_before_val = time.time()
    accuracy, error, false = getCAM(opts=opts, model=model, loader=val_loader, device=device)

    time_after_val = time.time()
    print('Time_val = %f' % (time_after_val - time_before_val))
    print("Accuracy:" + str(accuracy))
    print("Forget:" + str(error))
    print("Alarms:" + str(false))