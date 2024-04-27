import numpy as np
import os
import skimage.io as skio
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    src_path = './data/WDCD_dataset/data/label/'
    mask_path = './Validation/WDCD_GF1/cloudMap/'
    label_path = './Validation/WDCD_GF1/block_pre/'
    vis_path = './Validation/WDCD_GF1/back_oup_vis/'
    save_path = './Validation/WDCD_GF1/back_oup/'

    os.makedirs(vis_path,exist_ok=True)
    os.makedirs(save_path,exist_ok=True)

    num,folder=0,0

    cut_h = 250
    cut_w = 250
    ol = 125

    # get all origin tiff names
    thumb_list = []
    for t in os.listdir(src_path):
        content = t.replace('.tiff','').split('_')
        if content[0] not in thumb_list:
            thumb_list.append(content[0])
    print(thumb_list)

    for thumb in thumb_list:
        rsData = skio.imread(src_path + thumb + '.tiff',plugin="tifffile")
        src_h,src_w = rsData.shape[0],rsData.shape[1]
        backmap = torch.zeros((src_h,src_w,2),dtype=torch.float)

        folder+=1

        if os.path.exists(save_path + str(thumb) + '_back.npy'):
            continue

        for files in os.listdir(mask_path):
            content = files.replace('_4cam.npy','').split('_')

            if content[0] != thumb:
                continue

            print(str(folder)+'/'+str(num)+'----------------------------------Producing ' + files + '----------------------------------')
            num+=1

            pred_label = np.load(label_path + files.replace('_4cam.npy','') +'_2bl.npy')
            if np.sum(pred_label)==0:
                print('This block is predicted as clear. SKIP!!!')
                continue

            crop_data = np.load(mask_path + files)
            crop_data = torch.tensor(crop_data)

            crop_data = crop_data.to(device)
            backmap = backmap.to(device)

            Lc = int(content[1])
            Rc = int(content[2])
            Ur = int(content[3])
            Dr = int(content[4])
            Mc = Lc + ol
            Mr = Ur + ol

            if Rc-Lc<=ol or Dr-Ur<=ol:
                continue

            # block1   block2
            # block3   block4
            print('Source   : %sx%s' % (str(src_h),str(src_w)))
            print('Position : %sx%sx%s' % (str(Ur),str(Mr),str(Dr)))
            print('Position : %sx%sx%s' % (str(Lc),str(Mc),str(Rc)))
            print()

            if (Dr-Ur==cut_h) and (Rc-Lc==cut_w):
                backmap[Ur:Dr, Lc:Rc,0] = backmap[Ur:Dr, Lc:Rc,0] + crop_data
                temp = torch.ones((crop_data.shape[0],crop_data.shape[1]),dtype=torch.float)
                temp = temp.to(device)
                backmap[Ur:Dr, Lc:Rc,1] = backmap[Ur:Dr, Lc:Rc,1] + temp

        index = backmap[:-cut_h, :-cut_w, 1]

        joint = backmap[:-cut_h, :-cut_w, 0] / index
        joint = joint.cpu().numpy()
        np.save(save_path + str(thumb) + '_back.npy', joint)

        fig, ax = plt.subplots()
        ax.imshow(joint, cmap='gray', aspect='equal')
        plt.axis('off')
        height, width = joint.shape
        fig.set_size_inches(width / 100.0, height / 100.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)

        plt.savefig(vis_path + str(thumb) + '_back.png')
        plt.close()


