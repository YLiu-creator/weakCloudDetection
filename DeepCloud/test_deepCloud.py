from tqdm import tqdm
import network
import os
import random
import argparse
import numpy as np
from torch.utils import data
from datasets import deepcloud_test_gf1
from metrics import StreamSegMetrics
import torch
import torch.nn as nn
import time
from utils import ext_transforms as et


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--dataset", type=str, default='gf1', help='Name of dataset')
    parser.add_argument("--data_root", type=str, default='./GF1_datasets/data/',
                        help="path to Dataset")

    parser.add_argument("--gpu_id", type=str, default='4,5',help="GPU ID")
    parser.add_argument("--batch_size", type=int, default=8,help='batch size (default: 4)')
    parser.add_argument("--num_classes", type=int, default=2,help="num classes (default: None)")

    # Model Options
    parser.add_argument("--model", type=str, default='deepCloud', help='model name')

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=True)
    parser.add_argument("--ckpt", default='./checkpoints/deepCloud_light/best_epoch_deepCloud.pth',
                        help="restore from checkpoint")

    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"predict_path\"")
    parser.add_argument("--predict_path", default='./weakly_spuervisied_CD/PHCNet/',
                        help="save prediction results")

    parser.add_argument("--random_seed", type=int, default=1,help="random seed (default: 1)")
    return parser

def get_dataset(opts):
    if opts.dataset == 'gf1':
        val_dst = deepcloud_test_gf1(root=opts.data_root, image_set='test')

    if opts.dataset == 'WDCD':
        val_dst = wscd_test_wdcd(root=opts.data_root, image_set='test')

    return  val_dst



def validate(opts, model, loader, metrics, device, threshold):
    metrics.reset()

    index = 0

    with torch.no_grad():
        for i,(images, targets) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = targets.cpu().numpy()


            metrics.update(targets, preds)

            if opts.save_val_results:
                sample_fname = loader.sampler.data_source.masks[:]
                os.makedirs(opts.predict_path, exist_ok=True)
                os.makedirs(opts.predict_path + 'img/', exist_ok=True)
                os.makedirs(opts.predict_path + 'oup/', exist_ok=True)
                os.makedirs(opts.predict_path + 'tar/', exist_ok=True)
                os.makedirs(opts.predict_path + 'CAM/', exist_ok=True)
                # print('Save position is %s\n' % (opts.predict_path))

                for batch in range(images.shape[0]):
                    content = sample_fname[index].replace('.npy', "").split("/")
                    name = content[-1]
                    # print('%d ---------%s---------' % (index, name))
                    tiff = images.cpu().numpy()
                    np.save(opts.predict_path + 'img/' + name + '_0img.npy', tiff[batch, :, :, :])
                    np.save(opts.predict_path + 'tar/' + name + '_1tar.npy', targets[batch, :, :])
                    np.save(opts.predict_path + 'oup/' + name + '_2oup.npy', preds[batch, :, :])
                    np.save(opts.predict_path + 'CAM/' + name + '_2cam.npy', outputs[batch, :, :])
                    index = index + 1

        score = metrics.get_results()
    return score,threshold


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

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    val_dst = get_dataset(opts)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size, shuffle=False, num_workers=8, drop_last=True,
                                 pin_memory=False)

    print("Dataset: %s, val set: %d" % (opts.dataset, len(val_dst)))

    # Set up model
    print('Model = %s, num_classes=%d' % (opts.model, opts.num_classes))
    model = network.deepCloud_light()
    print(model)

    # Restore
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Model restored from %s" % opts.ckpt)
    else:
        print("Error: Can not load best checkpoints.")

    if opts.test_only:
        model.eval()

        for thres in [0.5]:
            print('************************************************************************')
            print(thres)
            time_before_val = time.time()

            val_score,threshold = validate(opts=opts, model=model, loader=val_loader,
                                           metrics=metrics, device=device,threshold=thres)

            time_after_val = time.time()
            print('Time_val = %f' % (time_after_val - time_before_val))
            print('Threshold = %f' % (threshold))
            print(metrics.to_str(val_score))