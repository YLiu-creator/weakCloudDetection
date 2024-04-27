import numpy as np
import os,cv2
import matplotlib.pyplot as plt
import math

base_path = '/home/liuyang/weakly_spuervisied_CD/Validation/WDCD_vgg16_paper/train/cloudMap/'
block_path = '/home/liuyang/weakly_spuervisied_CD/Validation/WDCD_vgg16_paper/train/block_label/'


num = 0
cloud,noncloud =0,0
mean,std = [],[]
max,min = [],[]

num_nan = 0


def normalization(data,max,min):
    _range = max-min
    return (data-min)/_range

for samples in os.listdir(base_path):

        name = samples.replace('_3cloudMap.npy', '')

        block_label = np.load(block_path + name+'_5bl.npy')

        num +=1
        print(str(num) + ' Producing ' + samples)

        # if math.isnan(np.mean(data)):
        #         num_nan+=1
        #         continue

        if block_label==0:
            noncloud+=1

        data = np.load(base_path + samples)
        print(data.shape)

        # data = normalization(data,max=30502212,min=-6478209.0)

        mean.append(np.mean(data))
        std.append(np.std(data))
        max.append(np.max(data))
        min.append(np.min(data))


all_mean = np.mean(mean)
all_std = np.mean(std)
all_max = np.max(max)
all_min = np.min(min)


print(all_mean)
print(all_std)
print(all_max)
print(all_min)
print('ALL files:%d'%(num))
print('noncloud files:%d'%(noncloud))