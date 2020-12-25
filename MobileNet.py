# coding: utf-8
import torch as t
from torch import nn
from torch.nn import functional as F

class DepthWiseNet(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(DepthWiseNet,self).__init__()
        self.depthwise=nn.Sequential(nn.Conv2d(inchannel,outchannel,3,1,1,bias=False),
                                    nn.BatchNorm2d(outchannel),
                                    nn.ReLU())
    def forward(self,x):
        return self.depthwise(x)
    
class PointWiseNet(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1):
        super(PointWiseNet,self).__init__()
        self.pointwise=nn.Sequential(nn.Conv2d(inchannel,outchannel,1,stride,0,bias=False),
                                    nn.BatchNorm2d(outchannel),
                                    nn.ReLU())
    def forward(self,x):
        return self.pointwise(x)
    
                                    
                                    
class MobileBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride):
        super(MobileBlock,self).__init__()
        self.DW=[]
        for i in range(inchannel):
            self.DW.append(DepthWiseNet(1,1))
            
        self.PW=PointWiseNet(inchannel,outchannel,stride)
        
    def Splitdata(self,x):
        data=[]
        for e in x[0]:
            data.append(e.unsqueeze(0))
        return data
    
    def Catdata(self,x):
        for i in range(len(x)):
            x[i]=self.DW[i](x[i].unsqueeze(0)).squeeze(0)
            x[i]=x[i].squeeze(0)
        data=t.stack(x)
        return data.unsqueeze(0)
    
    def forward(self,x):
        x=self.Splitdata(x)
        x=self.Catdata(x)
        x=self.PW(x)
        return x
        
        
class MobileNet(nn.Module):
    def __init__(self,nums_class=100):
        super(MobileNet,self).__init__()
        self.layer0=nn.Sequential(nn.Conv2d(3,32,3,2,1,bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU())
        self.layer1=self.make_layer(32,64,1,1)
        self.layer2=self.make_layer(64,128,2,1)
        self.layer3=self.make_layer(128,128,1,1)
        self.layer4=self.make_layer(128,256,2,1)
        self.layer5=self.make_layer(256,256,1,1)
        self.layer6=self.make_layer(256,512,2,1)
        self.layer7=self.make_layer(512,512,1,5)
        self.layer8=self.make_layer(512,1024,2,1)
        self.layer9=self.make_layer(1024,1024,1,1)
        
        self.fc=nn.Linear(1024,nums_class)
        
    def make_layer(self,inchannel,outchannel,stride,num_block):
        layers=[]
        for i in range(num_block):
            layers.append(MobileBlock(inchannel,outchannel,stride))
        
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x=self.layer0(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        x=self.layer6(x)
        x=self.layer7(x)
        x=self.layer8(x)
        x=self.layer9(x)
        
        x=F.avg_pool2d(x,7)
        x=x.view(x.size(0),-1)
        
        return self.fc(x)

