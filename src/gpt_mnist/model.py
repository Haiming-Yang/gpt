import torch
import torch.nn as nn
import torchvision.models as mod

class ResGPTNet34(nn.Module):
    def __init__(self, nG0, Nj, output_mode=None):
        super(ResGPTNet34, self).__init__()

        self.output_mode = output_mode

        self.tracker = {'iter':0}
        self.setting = {'softened_learning_rate': False}

        self.n_class = 10
        self.channel_adj = nn.Conv2d(1,3,1, bias=False)
        self.channel_adj.weight.data = self.channel_adj.weight.data*0+1
        self.channel_adj.requires_grad = False

        self.nG0 = nG0
        self.Nj = Nj
        self.backbone = mod.resnet34(pretrained=True)

        self.fc = nn.Conv2d(512,self.n_class,1,bias=False)

        # Extra conv block
        self.cf = ConvPipe(3,64)
        self.cout1 = ConvPipe(128,64) # unused
        self.cyg = ConvPipe(128,self.nG0+1)
        self.cys = ConvPipe(128,self.Nj)

        # Expansion path
        self.eb2 = ExpansionBlock(128,64,k=3, stride=2)
        self.eb1 = ExpansionBlock(128,64,k=3, stride=2)
        self.eb0 = ExpansionBlock(128,64,k=3, stride=2)


    def forward(self, x):
        batch_size = x.shape[0]
        x = self.channel_adj(x) # 3, 28,28
        x0 = self.cf(x) # 64,28,28
        # print('x0.shape:',x0.shape)

        ########################################
        # preserve the ResNet structure here
        mc = self.backbone.children() # this returns an iterator

        # Resnet
        x1 = next(mc)(x) # conv1
        x1 = next(mc)(x1) # bn1
        x1 = next(mc)(x1) # relu

        x2 = next(mc)(x1) # shape (batch,64,14,14) (maxpool)
        x2 = next(mc)(x2) # shape (batch,64,7,7) (Resnet L1)

        x3 = next(mc)(x2) # shape (batch,128,4,4) (Resnet L2)
        x4 = next(mc)(x3) # shape (batch,256,2,2) (Resnet L3)
        x5 = next(mc)(x4) # shape (batch,512,1,1) (Resnet L4)
        ########################################

        y = self.fc(x5).reshape(batch_size,self.n_class)
        if self.output_mode=='prediction_only':
            return y
    
        v2 = torch.cat((x2,self.eb2(x3)[:,:,1:-1,1:-1]),dim=1) # 128,7,7
        v1 = torch.cat((x1,self.eb1(v2)[:,:,:-1,:-1]),dim=1) # 128,14,14
        v0 = torch.cat((x0,self.eb0(v1)[:,:,:-1,:-1]),dim=1) # 128,28,28)

        yg = self.cyg(v0)
        ys = self.cys(v0)

        return y, yg, ys

class ExpansionBlock(nn.Module):
    def __init__(self, input_channels, output_channels, k=3, stride=1,dilation=1):
        super(ExpansionBlock, self).__init__()
    
        self.c1 = nn.ConvTranspose2d(input_channels,output_channels,k,stride=stride,dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.c1(x)
        out = self.bn1(out)
        out = self.act(out)
        return out        

class ConvPipe(nn.Module):
    """
    Just a simple convolution block to mimic the first conv layers with the same size as the input
    """
    def __init__(self, input_channels, output_channels):
        super(ConvPipe, self).__init__()

        self.c1 = nn.Conv2d(input_channels, output_channels,3,bias=False)
        self.c2 = nn.ConvTranspose2d(output_channels,output_channels,3, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.act(self.c1(x))
        x = self.act(self.c2(x))
        return x

        

        