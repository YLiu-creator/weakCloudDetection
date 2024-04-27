from tqdm import tqdm
import network
import os
import random
import argparse
import numpy as np
from torch.utils import data
from datasets import GF1_WSCD_valid_cls,GF1_WSCD_train_cls,WDCD_test_cls
import torch
import torch.nn as nn
import time

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--dataset", type=str, default='GF-1', help='Name of dataset')
    parser.add_argument("--data_root", type=str, default='./GF1_datasets/data',
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
    parser.add_argument("--test_only", action='store_true', default=True)
    parser.add_argument("--ckpt", default='./checkpoints/WDCD_vgg16_GF1/best_...',
                        help="restore from checkpoint")

    parser.add_argument("--save_val_results", action='store_true', default=True,
                        help="save segmentation results to \"predict_path\"")
    parser.add_argument("--predict_path", default='./Validation/WDCD_GF1/',
                        help="save prediction results")

    parser.add_argument("--gpu_id", type=str, default='1',
                        help="GPU ID")

    parser.add_argument("--batch_size", type=int, default=2,
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
    if opts.dataset=='WDCD':
        val_dst = WDCD_test_cls(root=opts.data_root,image_set='test',input_form='TIFF')
    if opts.dataset=='GF1':
        val_dst = GF1_WSCD_valid_cls(root=opts.data_root,image_set='test',input_form='TIFF')

    return val_dst


def validate(opts, model, loader, device):
    # metrics.reset()

    index = 0
    sample_fname = loader.sampler.data_source.images[:]
    with torch.no_grad():
        for i,(images, block_label,target) in tqdm(enumerate(loader)):
            # print(str(i) + '----------------------------------------------')
            images = images.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.long)
            block_label = block_label.to(device, dtype=torch.long)

            oup,cls = model(images)  # 4,1024,320,320
            cloudMap = oup[:,1,:,:]
            block_pre = cls.detach().max(dim=1)[1].cpu().numpy()

            images = images.cpu().numpy()
            target = target.cpu().numpy()
            block_label = block_label.cpu().numpy()
            cloudMap = cloudMap.cpu().numpy()

            os.makedirs(opts.predict_path, exist_ok=True)
            os.makedirs(opts.predict_path + 'img/', exist_ok=True)
            os.makedirs(opts.predict_path + 'tar/', exist_ok=True)
            os.makedirs(opts.predict_path + 'block_label/', exist_ok=True)
            os.makedirs(opts.predict_path + 'block_pre/', exist_ok=True)
            os.makedirs(opts.predict_path + 'cloudMap/', exist_ok=True)
            # print('Save position is %s\n' % (opts.predict_path))

            for batch in range(images.shape[0]):
                content = sample_fname[index].replace('.tiff', "").split("/")

                name = content[-1]
                # print('%d ---------%s---------' % (index, name))
                np.save(opts.predict_path + 'img/' + name + '_0img.npy', images[batch, :, :, :])
                np.save(opts.predict_path + 'tar/' + name + '_1tar.npy', target[batch, :, :])
                np.save(opts.predict_path + 'block_label/' + name + '_2bl.npy', block_label[batch])
                np.save(opts.predict_path + 'block_pre/' + name + '_3pre.npy', block_pre[batch])
                np.save(opts.predict_path + 'cloudMap/' + name + '_4cam.npy', cloudMap[batch, :, :])
                index = index + 1

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
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size, shuffle=False,num_workers=8,drop_last=True,pin_memory=False)
    print("Dataset: %s, Val set: %d" % (opts.dataset, len(val_dst)))

    # Set up model
    model_map = {
        'GF1_vgg16':network.GF1_VGG16PP,
    }

    print('Model = %s, num_classes=%d' % (opts.model, opts.num_classes))
    model = model_map[opts.model](num_classes=opts.num_classes)

    print(model)

    # Restore
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint["model_state"].items() if (k in model_dict)}
        b,c = model_dict["fc1.weight"].size(0),model_dict["fc1.weight"].size(1)
        pretrained_dict["fc1.weight"] = checkpoint["model_state"]["fc1.weight"].view((b,c,1,1))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model = nn.DataParallel(model)
        model.to(device)
        print("Model restored from %s" % opts.ckpt)
    else:
        print("Error: Can not load best checkpoints.")


    if opts.test_only:
        model.eval()
        time_before_val = time.time()

        validate(opts=opts, model=model, loader=val_loader, device=device)

        time_after_val = time.time()
        # print(metrics.to_str(val_score))
        print('Time_val = %f' % (time_after_val - time_before_val))

if __name__ == '__main__':
    main()