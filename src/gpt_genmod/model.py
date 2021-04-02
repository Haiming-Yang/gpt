import numpy as np

import torch
import torch.nn as nn
# import torch.nn.functional as F

class SimpleGrowth(nn.Module):
    def __init__(self, hidden_layer=4, **kwargs):
        super(SimpleGrowth, self).__init__()
        self.total_iter = 0
        self.best_avg_loss = np.inf
        self.OBSERVE = kwargs['OBSERVE']
        
        C, H, W = (3,32,32) if kwargs['img_shape'] is None else kwargs['img_shape']

        self.D = 15 # gen's number of features
        self.n_channel = 96

        encoder_decoder_args = {
            'D': self.D,
            'n_channel': self.n_channel,
            'img_shape': (C,H,W),
        }

        self.conv1 = get_one_funnel_conv(C, self.n_channel, self.D, bias=False, group_all=False, groups=3)

        encoder_module_list = []
        for i in range(hidden_layer):
            encoder_module_list.append(PatternEncode(**encoder_decoder_args))
        self.en = nn.Sequential(*encoder_module_list)
        
        n_hidden = int(self.D * 4 * 4)
        self.fc = nn.Linear(n_hidden,n_hidden,bias=False) 
        self.fc.weight.data = self.fc.weight.data*0
        for i in range(n_hidden):
            self.fc.weight.data[i,i]  = 1.
        self.act = nn.Tanh()

        decoder_module_list = []
        for i in range(hidden_layer):
            decoder_module_list.append(PatternGrowth(**encoder_decoder_args))
        
        # decoder_module_list.append(nn.ConvTranspose2d(self.D, 3,3,bias=False))
        self.de = nn.Sequential(*decoder_module_list)


    def forward(self,x):
        # x = self.encoder(x) 
        x = self.conv1(x)
        x = self.en(x)
        
        b, C, H, W = x.shape
        x = x.reshape(b,-1)

        if self.OBSERVE:
            print('encoded x.shape:', x.shape) 
            print('end observation.')
            exit()

        x = self.fc(x) 
        x = self.act(x)  + 0.1*torch.rand_like(x)

        x = x.reshape(b,C,H,W)
        x = self.de(x)

        x = torch.sigmoid(x[:,:3,:,:])
        return x

    def sample(self, latent_img_shape, latent_vec):
        b,C,H,W = latent_img_shape

        x = latent_vec

        x = self.fc(x) 
        x = self.act(x)  + 0.1*torch.rand_like(x)

        x = x.reshape(b,C,H,W)
        x = self.de(x)

        x = torch.sigmoid(x[:,:3,:,:])
        return x

###################################################
# The components
###################################################        


def get_one_funnel_conv(input_channel, mid_channel, out_channel, group_all=False,  bias=False, groups=3):
    if group_all:
        assert(input_channel==mid_channel)
        assert(mid_channel==out_channel)
        groups=mid_channel

    module_list = [nn.ConvTranspose2d(input_channel,mid_channel, 3, bias=bias, groups=groups),nn.Tanh(),]
    module_list = module_list + [nn.Conv2d(mid_channel,out_channel, 3, bias=bias, groups=groups),nn.Tanh(),]
    seq =  nn.Sequential(*module_list)
    return seq 


class PatternGrowth(nn.Module):
    def __init__(self, **kwargs):
        super(PatternGrowth, self).__init__()

        D = kwargs['D']
        bias = False
        n_channel = kwargs['n_channel']
        
        self.D = D
        self.DTYPE = 3 # 3 features from self.D to determine the TYPE of generators
        # self.topology = np.array([[0,1],[1,0],[0,-1],[-1,0]]) # TOP, RIGHT, BOTTOM, LEFT
        self.topology =np.array([[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]]) 
        # self.topology = np.array([[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1],
        #     [0,2],[2,0],[0,-2],[-2,0]]) 
        self.Nj = len(self.topology)


        ce_modules = [
            nn.Conv2d(self.D,n_channel, 1, bias=bias), nn.BatchNorm2d(n_channel),nn.Tanh(), 
            nn.Conv2d(n_channel, 1, 1, bias=bias),]
        self.compenv_conv = nn.Sequential(*ce_modules)

        cdiv_modules = [
            nn.ConvTranspose2d( self.D, n_channel, 2, stride=2, bias=bias, groups=3), nn.BatchNorm2d(n_channel), nn.Tanh(),
            get_one_funnel_conv(n_channel,n_channel,self.D), ] 
        self.cell_div = nn.Sequential(*cdiv_modules)

        born_modules = [
            nn.ConvTranspose2d(self.D, n_channel, 1, bias=bias, ), nn.BatchNorm2d(n_channel), nn.Tanh(),
            nn.ConvTranspose2d(n_channel, self.D, 1, bias=bias, ),]
        self.born = nn.Sequential(*born_modules)


        divdet_modules = [  
            get_one_funnel_conv(self.Nj + self.DTYPE,n_channel,n_channel, groups=1), nn.BatchNorm2d(n_channel), nn.Tanh(),
            nn.ConvTranspose2d( n_channel, 1, 2, stride=2, bias=bias), nn.Sigmoid(),]
        self.div_change_determinant = nn.Sequential(*divdet_modules)

    def compenv(self, x):
        J = self.topology

        batch_size, c, h, w = x.shape
        ENV = torch.zeros(batch_size,self.Nj, h, w).to(device=x.device)

        for k in range(self.Nj):
            roll_right = J[k,0]
            roll_down =  -J[k,1]
            temp = torch.roll(x, int(roll_right), 1+2) # roll right by roll_right
            temp = torch.roll(temp,int(roll_down),0+2) # roll down by roll_down
            
            ENV[:,k:k+1,:,:] = self.compenv_conv(temp)
        return ENV

    def growth(self, x, ENV, div=True):
        type_and_env = torch.cat((x[:,:self.DTYPE,:,:], ENV), dim=1)

        CHANGE = self.div_change_determinant(type_and_env)
        x = self.cell_div(x)

        born = self.born(x)

        x = x*(1-CHANGE) + CHANGE * born
        return x


    def forward(self,x):
        ENV = self.compenv(x)
        x = self.growth(x, ENV)
        return x



class PatternEncode(nn.Module):
    def __init__(self, **kwargs):
        super(PatternEncode, self).__init__()

        D = kwargs['D']
        bias = False
        n_channel = kwargs['n_channel']
        
        self.D = D
        self.DTYPE = 2 #  3 features from self.D to determine the TYPE of generators
        self.topology =np.array([[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]]) 
        self.Nj = len(self.topology)

        ce_modules = [
            nn.Conv2d(self.D,n_channel, 1, bias=bias), nn.BatchNorm2d(n_channel),nn.Tanh(), 
            nn.Conv2d(n_channel, 1, 1, bias=bias),]
        self.compenv_conv = nn.Sequential(*ce_modules)

        cmerge_modules = [
            nn.Conv2d( self.D, n_channel, 2, stride=2, bias=bias, groups=3), nn.BatchNorm2d(n_channel), nn.Tanh(),
            get_one_funnel_conv(n_channel,n_channel,self.D),] 
        self.cell_merge = nn.Sequential(*cmerge_modules)

        born_modules = [
            nn.Conv2d(self.D, n_channel, 1, bias=bias, ), nn.BatchNorm2d(n_channel), nn.Tanh(),
            nn.Conv2d(n_channel, self.D, 1, bias=bias, ),]
        self.born = nn.Sequential(*born_modules)

        mergedet_modules = [
            get_one_funnel_conv(self.Nj + self.DTYPE,n_channel,n_channel, groups=1), nn.BatchNorm2d(n_channel), nn.Tanh(),
            nn.Conv2d( n_channel, 1, 2, stride=2, bias=bias), nn.Sigmoid(),]
        self.merge_change_determinant = nn.Sequential(*mergedet_modules)

    def compenv(self, x):
        J = self.topology

        batch_size, c, h, w = x.shape
        ENV = torch.zeros(batch_size,self.Nj, h, w).to(device=x.device)

        for k in range(self.Nj):
            roll_right = J[k,0]
            roll_down =  -J[k,1]
            temp = torch.roll(x, int(roll_right), 1+2) # roll right by roll_right
            temp = torch.roll(temp,int(roll_down),0+2) # roll down by roll_down
            
            ENV[:,k:k+1,:,:] = self.compenv_conv(temp)
        return ENV

    def growth(self, x, ENV, div=True):
        type_and_env = torch.cat((x[:,:self.DTYPE,:,:], ENV), dim=1)

        CHANGE = self.merge_change_determinant(type_and_env)
        x = self.cell_merge(x)

        born = self.born(x)

        x = x*(1-CHANGE) + CHANGE * born
        x[:,:3,:,:] = torch.sigmoid(x[:,:3,:,:])
        return x


    def forward(self,x):
        ENV = self.compenv(x)
        x = self.growth(x, ENV)
        return x