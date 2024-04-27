from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from torch.utils import data
from datasets import WSFNet_stage2_segGF1_train,WSFNet_stage2_segGF1_test
import torch
import torch.nn as nn
import time
from metrics import StreamSegMetrics
import torch.nn.functional as F


def get_argparser():
    parser = argparse.ArgumentParser()

    # Save position
    parser.add_argument("--save_dir", type=str, default='./checkpoints/',
                        help="path to Dataset")

    # Datset Options
    parser.add_argument("--dataset", type=str, default='gf1', help='Name of dataset')
    parser.add_argument("--data_root", type=str, default='./GF1_datasets/data',
                        help="path to Dataset")
    parser.add_argument("--proposal_mask", type=str,
                        default='./weakly_spuervisied_CD/WSFNet/proposal_mak/',
                        help="path to Dataset")


    # Model Options
    parser.add_argument("--model", type=str, default='WSFNet_stage2_GF1',
                        choices=['WDCD_vgg16_cam','scExtractor','WSCD_resnet50','WSCD_resnet50_cam'], help='model name')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: None)")
    parser.add_argument("--in_channels", type=int, default=4,
                        help="num input channels (default: None)")
    parser.add_argument("--feature_scale", type=int, default=2,
                        help="feature_scale (default: 2)")
    parser.add_argument("--gpu_id", type=str, default='2',
                        help="GPU ID")

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--total_itrs", type=int, default=300000,
                        help="epoch number (default: 100k)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help='batch size (default: 4)')
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate (default: 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--optimizer_strtegy", type=str, default='Adam', choices=['Adam', 'SGD'],
                        help="Optimizer strtegies")
    parser.add_argument("--lr_policy", type=str, default='step', choices=['poly', 'step', 'checkUpdate'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'],
                        help="loss type (default: False)")


    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"predict_path\"")
    parser.add_argument("--predict_path", default='./weakly_spuervisied_CD/PHCNet/',
                        help="save prediction results")

    parser.add_argument("--continue_training", action='store_true', default=False)
    parser.add_argument("--ckpt", default=None,help="restore from checkpoint")

    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10, help="print interval of loss (default: 10)")

    return parser

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'gf1':
        train_dst = WSFNet_stage2_segGF1_train(root=opts.data_root, mask_dir=opts.proposal_mask, image_set='train')
        val_dst = WSFNet_stage2_segGF1_test(root=opts.data_root, image_set='test')
    return train_dst, val_dst


def validate(opts, model, loader, metrics, device):
    metrics.reset()

    index = 0

    with torch.no_grad():
        for i, (images, targets) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.long)

            cls_pre, seg_pre = model(images)

            seg_pre = F.interpolate(seg_pre, size=(int(320), int(320)), mode='bilinear', align_corners=False)
            preds = seg_pre.detach().max(dim=1)[1].cpu().numpy()

            targets = targets.cpu().numpy()

            metrics.update(targets, preds)

            if opts.save_val_results:
                sample_fname = loader.sampler.data_source.masks[:]
                os.makedirs(opts.predict_path, exist_ok=True)
                os.makedirs(opts.predict_path + 'img/', exist_ok=True)
                os.makedirs(opts.predict_path + 'oup/', exist_ok=True)
                os.makedirs(opts.predict_path + 'tar/', exist_ok=True)
                for batch in range(images.shape[0]):
                    content = sample_fname[index].replace('.npy', "").split("/")
                    name = content[-1]
                    tiff = images.cpu().numpy()
                    np.save(opts.predict_path + 'img/' + name + '_0img.npy', tiff[batch, :, :, :])
                    np.save(opts.predict_path + 'tar/' + name + '_1tar.npy', targets[batch, :, :])
                    np.save(opts.predict_path + 'oup/' + name + '_2oup.npy', preds[batch, :, :])
                    index = index + 1

        score = metrics.get_results()
    return score


def main():

    opts = get_argparser().parse_args()

    save_dir = os.path.join(opts.save_dir + opts.model + '/')
    os.makedirs(save_dir, exist_ok=True)
    print('Save position is %s\n'%(save_dir))

    # select the GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s,  CUDA_VISIBLE_DEVICES: %s\n" % (device, opts.gpu_id))


    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)


    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=128,
                                   drop_last=True,pin_memory=False)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size, shuffle=True, num_workers=128,
                                 drop_last=True,pin_memory=False)
    print("Dataset: %s, Train set: %d, Val set: %d" % (opts.dataset, len(train_dst), len(val_dst)))

    val_interval = int(len(train_dst) / (opts.batch_size*4))
    # val_interval = 10

    # Set up model
    model_map = {
        'WSFNet_stage2_GF1':network.FCN,
    }

    print('Model = %s, num_classes=%d' % (opts.model, opts.num_classes))
    model = model_map[opts.model]()

    print(model)

    # Set up optimizer_strtegy
    if opts.optimizer_strtegy == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    elif opts.optimizer_strtegy == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.9)


    # Set up criterion
    if opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()


    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_epochs": cur_epochs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_loss": train_epoch_loss,
        }, path)
        print("Model saved as %s\n\n" % path)

    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint["model_state"].items() if (k in model_dict)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            scheduler.max_iters=opts.total_itrs
            scheduler.min_lr= opts.lr
            cur_itrs = checkpoint["cur_epochs"] * opts.val_interval
            # best_score = checkpoint['best_score']
            print("Continue training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        print("Best_score is %s" % (str(best_score)))
        # del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#

    interval_loss = 0
    train_epoch_best_loss = 99999999
    train_loss = list()
    no_optim = 0
    best_epoch = 1
    learning_rate = []
    train_accuracy = list()
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        train_epoch_loss= 0

        for (images, target) in train_loader:
            if (cur_itrs)==0 or (cur_itrs) % opts.print_interval == 0:
                t1 = time.time()

            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs, cam = model(images)

            preds = F.interpolate(cam, size=(int(320), int(320)), mode='bilinear', align_corners=False)

            loss = criterion(preds, target)

            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            train_epoch_loss += np_loss

            if (cur_itrs) % opts.print_interval == 0:
                interval_loss = interval_loss / opts.print_interval
                train_loss.append(interval_loss)
                t2 = time.time()

                lr_print = scheduler.get_lr()[0]
                if lr_print not in learning_rate:
                    learning_rate.append(lr_print)

                print("Epoch %d, Itrs %d/%d, Loss=%f (%f/%f), Learning rate = %f, Time = %f" %
                      (cur_epochs,cur_itrs,opts.total_itrs,interval_loss,train_epoch_loss,train_epoch_best_loss,lr_print,t2-t1,))
                interval_loss = 0.0

             # save the ckpt file per 5000 itrs
            if (cur_itrs) % val_interval == 0:
                print("validation...")
                model.eval()

                time_before_val = time.time()

                val_score = validate(opts=opts, model=model, loader=val_loader, metrics=metrics,device=device)

                time_after_val = time.time()
                print('Time_val = %f' % (time_after_val - time_before_val))
                print(metrics.to_str(val_score))

                if val_score['F_score'] > best_score:  # save best model
                    no_optim = 0
                    best_score = val_score['F_score']
                    save_ckpt(save_dir + 'best_epoch%s_%s_%s.pth' % (str(cur_epochs), opts.model, opts.dataset))
                else:
                    no_optim += 1
                    save_ckpt(save_dir + 'latest_epoch%s_%s_%s.pth' % (str(cur_epochs), opts.model, opts.dataset))

                model.train()

        if cur_itrs >= opts.total_itrs:
                print(cur_itrs)
                print(opts.total_itrs)
                return

        print('Update learning rate')
        scheduler.step()  # update

        if no_optim > 20:
            print('Early stop at %d epoch' % cur_epochs)
            print('Best epoch is %d'% best_epoch)
            break

    print('Best epoch is %d' % best_epoch)

if __name__ == '__main__':
    main()
