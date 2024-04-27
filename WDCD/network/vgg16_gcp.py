
from typing_extensions import final
import torch
from torch import nn
import torch.nn.functional as F

eps = 1e-08
final_depth = 1024 

class vgg_16_gcp(nn.Module):
    def __init__(self):
        super(vgg_16_gcp,self).__init__()
        self.conv1=nn.Conv2d(4,64,3)
        self.conv2=nn.Conv2d(64,64,3)
        self.conv3=nn.Conv2d(64,128,3)
        self.conv4=nn.Conv2d(128,128,3)

        self.conv5=nn.Conv2d(128,256,3)
        self.conv6=nn.Conv2d(256,256,3)
        self.conv7=nn.Conv2d(256,256,3)

        self.conv8=nn.Conv2d(256,512,3)
        self.conv9=nn.Conv2d(512,512,3)
        self.conv10=nn.Conv2d(512,final_depth,3)

        self.conv11=nn.Conv2d(final_depth,1,20) # GCP, feature size 20*20

        self.fc1=nn.Linear(final_depth,2)


    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.max_pool2d(x,2)

        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        x=F.max_pool2d(x,2)

        x=F.relu(self.conv5(x))
        x=F.relu(self.conv6(x))
        x=F.relu(self.conv7(x))
        x=F.max_pool2d(x,2)

        x=F.relu(self.conv8(x))
        x=F.relu(self.conv9(x))
        x=F.relu(self.conv10(x))

        weight=self.conv11.weight
        x=x*weight
        x=torch.sum(torch.sum(x,dim=-1),dim=-1)

        x=x.view(x.shape[0],-1)
        x=self.fc1(x)
        return x

class vgg_16_gcp_cam_nopool_mean(nn.Module):
    def __init__(self):
        super(vgg_16_gcp_cam_nopool_mean,self).__init__()
        self.conv1=nn.Conv2d(4,64,3)
        self.conv2=nn.Conv2d(64,64,3)

        self.conv3=nn.Conv2d(64,128,3)
        self.conv4=nn.Conv2d(128,128,3)

        self.conv5=nn.Conv2d(128,256,3)
        self.conv6=nn.Conv2d(256,256,3)
        self.conv7=nn.Conv2d(256,256,3)

        self.conv8=nn.Conv2d(256,512,3)
        self.conv9=nn.Conv2d(512,512,3)
        self.conv10=nn.Conv2d(512,final_depth,3)
        self.conv11=nn.Conv2d(final_depth,1,20) # GCP, feature size 20*20

        self.fc1=nn.Conv2d(final_depth,2,1)

    def forward(self,x):
        y=F.relu(self.conv1(x))
        y=F.relu(self.conv2(y))
        y=F.max_pool2d(y,2)
        y=F.relu(self.conv3(y))
        y=F.relu(self.conv4(y))
        y=F.max_pool2d(y,2)
        y=F.relu(self.conv5(y))
        y=F.relu(self.conv6(y))
        y=F.relu(self.conv7(y))
        y=F.max_pool2d(y,2)
        y=F.relu(self.conv8(y))
        y=F.relu(self.conv9(y))
        y=F.relu(self.conv10(y))
        weight=self.conv11.weight
        a=y*weight
        a=torch.sum(torch.sum(a,dim=-1),dim=-1)
        a = a.expand(230,230,final_depth)
        a = torch.unsqueeze(a,dim=-1)
        a = a.permute(3,2,1,0)

        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))

        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))

        x=F.relu(self.conv5(x))
        x=F.relu(self.conv6(x))
        x=F.relu(self.conv7(x))

        x=F.relu(self.conv8(x))
        x=F.relu(self.conv9(x))
        x=F.relu(self.conv10(x))

        mean = torch.mean(torch.mean(x,dim=-1),dim=-1)
        mean = mean.expand(230,230,final_depth)
        mean = torch.unsqueeze(mean,dim=-1)
        mean = mean.permute(3,2,1,0)
        x = x*a/(mean+eps)

        x=self.fc1(x)
        return x

class vgg_16_gcp_cam_nopool_median(nn.Module):
    def __init__(self):
        super(vgg_16_gcp_cam_nopool_median,self).__init__()
        self.conv1=nn.Conv2d(4,64,3)
        self.conv2=nn.Conv2d(64,64,3)

        self.conv3=nn.Conv2d(64,128,3)
        self.conv4=nn.Conv2d(128,128,3)

        self.conv5=nn.Conv2d(128,256,3)
        self.conv6=nn.Conv2d(256,256,3)
        self.conv7=nn.Conv2d(256,256,3)

        self.conv8=nn.Conv2d(256,512,3)
        self.conv9=nn.Conv2d(512,512,3)
        self.conv10=nn.Conv2d(512,final_depth,3)
        self.conv11=nn.Conv2d(final_depth,1,20)

        self.fc1=nn.Conv2d(final_depth,2,1)

    def forward(self,x):
        y=F.relu(self.conv1(x))
        y=F.relu(self.conv2(y))
        y=F.max_pool2d(y,2)
        y=F.relu(self.conv3(y))
        y=F.relu(self.conv4(y))
        y=F.max_pool2d(y,2)
        y=F.relu(self.conv5(y))
        y=F.relu(self.conv6(y))
        y=F.relu(self.conv7(y))
        y=F.max_pool2d(y,2)
        y=F.relu(self.conv8(y))
        y=F.relu(self.conv9(y))
        y=F.relu(self.conv10(y))
        weight=self.conv11.weight
        a=y*weight
        a=torch.sum(torch.sum(a,dim=-1),dim=-1)
        a = a.expand(230,230,final_depth)
        a = torch.unsqueeze(a,dim=-1)
        a = a.permute(3,2,1,0)

        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))

        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))

        x=F.relu(self.conv5(x))
        x=F.relu(self.conv6(x))
        x=F.relu(self.conv7(x))

        x=F.relu(self.conv8(x))
        x=F.relu(self.conv9(x))
        x=F.relu(self.conv10(x))

        mymedian = torch.median(torch.median(x,dim=-1).values,dim=-1).values
        mymedian = mymedian.expand(230,230,final_depth)
        mymedian = torch.unsqueeze(mymedian,dim=-1)
        mymedian = mymedian.permute(3,2,1,0)
        x = x*a/(mymedian+eps)

        x=self.fc1(x)
        return x