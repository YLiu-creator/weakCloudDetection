import os
import argparse
import numpy as np
from metrics import StreamSegMetrics
import cv2

import matplotlib.pyplot as plt
import time
import math

def get_argparser():
    parser = argparse.ArgumentParser()

    # Test options
    parser.add_argument("--test_only", action='store_true', default=True)

    # Save position
    parser.add_argument("--save_dir", type=str, default='./Validation/WDCD_GF1/',
                        help="path to Dataset")
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")

    # Datset Options
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: 2)")
    parser.add_argument("--label_path", type=str,
                        default='./data/WDCD_dataset/data/label/',
                        help="path to Dataset txt file")
    parser.add_argument("--output_path", type=str,
                        default='./Validation/WDCD_GF1/back_oup/',
                        help="path to Dataset txt file")
    parser.add_argument("--save_path", type=str,
                        default='./Validation/WDCD_GF1/cloud_detection/',
                        help="path to Dataset txt file")

    parser.add_argument("--dst_mean", type=int, default=2,
                        help="the cam mean")
    parser.add_argument("--dst_std", type=int, default=2,
                        help="the cam std")
    parser.add_argument("--dst_multi", type=int, default=2,
                        help="caculate the threshold")

    return parser

def normalization(data,max,min):
    _range = max-min
    return (data-min)/_range


def validate(opts, loader, metrics, mean, std, multi):

    """Do validation and return specified samples"""
    metrics.reset()
    num = 0

    if opts.save_val_results:
        save_results_dir = os.path.join(opts.save_dir + opts.model + '/')
        if not os.path.exists(save_results_dir):
            os.makedirs(save_results_dir)

    for sample in loader:
        num+=1
        print(str(num) + '      Producing ' + sample + '-' * 20)

        target = np.load(opts.label_path + sample)

        pred_mask = np.load(opts.output_path +sample.replace('_1tar.npy','_4cam.npy'))

        threshold = mean + multi*std

        print('Target shape:  %dx%d' % (target.shape[0],target.shape[1]))
        print('Predict shape: %dx%d' % (pred_mask.shape[0], pred_mask.shape[1]))
        output = np.zeros((pred_mask.shape[0], pred_mask.shape[1]), dtype=int)
        output[pred_mask < threshold] = 0
        output[pred_mask >= threshold] = 1

        np.save(opts.save_path + sample.replace('_1tar.npy','_3oup.npy'),output)

        metrics.update(target, output)

    score = metrics.get_results()
    return score


if __name__ == '__main__':
    opts = get_argparser().parse_args()

    test_dst = os.listdir(opts.label_path)

    os.makedirs(opts.save_path,exist_ok=True)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    if opts.test_only:
        time_before_val = time.time()

        val_score = validate(opts=opts, loader=test_dst, metrics=metrics,
                             mean=opts.dst_mean, std=opts.dst_std, multi=opts.dst_multi )

        time_after_val = time.time()

        print('Time_val = %f' % (time_after_val - time_before_val))
        print(metrics.to_str(val_score))
