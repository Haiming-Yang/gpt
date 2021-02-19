from .components import *
from ..gpt_mnist.model import ConvPipe

class AutoGPTNet(nn.Module):
    def __init__(self, num_blocks=12):
        super(AutoGPTNet, self).__init__()
        self.num_blocks = num_blocks

        self.channel_adj = nn.ConvTranspose2d(1,16,5, bias=False)
        self.channel_adj.weight.data = self.channel_adj.weight.data*0 + 1/25
        # print(self.channel_adj.weight.data)

        self.cb1 = BENASConvBlock(input_channel=16, num_blocks=num_blocks)
        self.cb2 = BENASConvBlock(input_channel=64, num_blocks=num_blocks)
        self.cb3 = BENASConvBlock(input_channel=64, num_blocks=num_blocks)

        self.db3 = BENASDeConvBlock(input_channel=64, num_blocks=num_blocks)
        self.db2 = BENASDeConvBlock(input_channel=128, num_blocks=num_blocks)
        self.db1 = BENASDeConvBlock(input_channel=128, num_blocks=num_blocks)

        self.conv_resize = nn.Conv2d(128,128,5)

        nG0, Nj = 3, 8

        self.cyg = ConvPipe(128,nG0+1)
        self.cys = ConvPipe(128,Nj)

    def forward(self, x, dags, verbose=0):
        x = self.channel_adj(x)

        x1, x = self.cb1.forward(x, self.num_blocks, 
            self.cb1.construct_dag(dags['cb1']['prev_nodes'], dags['cb1']['modules'], appendage=dags['cb1']['downsampler']))
        x2, x = self.cb2.forward(x, self.num_blocks, 
            self.cb2.construct_dag(dags['cb2']['prev_nodes'], dags['cb2']['modules'], appendage=dags['cb2']['downsampler']))
        x3, x = self.cb3.forward(x, self.num_blocks, 
            self.cb3.construct_dag(dags['cb3']['prev_nodes'], dags['cb3']['modules'], appendage=dags['cb3']['downsampler']))

        if verbose>=100:
            print('downsampling...')
            for a,b in zip(['x1','x2','x3','x'],[x1,x2,x3,x]):
                print('%s:%s'%(str(a),str(b.shape)))
            print('upsampling...')

        _ , x = self.db3.forward(x, self.num_blocks, 
            self.db3.construct_dag(dags['db3']['prev_nodes'], dags['db3']['modules'], appendage=dags['db3']['upsampler']))
        x = torch.cat((x,x3),axis=1)
        if verbose>=100: print('%s:%s'%(str('x'),str(x.shape)))

        _ , x = self.db2.forward(x, self.num_blocks, 
            self.db2.construct_dag(dags['db2']['prev_nodes'], dags['db2']['modules'], appendage=dags['db2']['upsampler']))
        x = torch.cat((x,x2),axis=1)
        if verbose>=100: print('%s:%s'%(str('x'),str(x.shape)))

        _ , x = self.db1.forward(x, self.num_blocks, 
            self.db1.construct_dag(dags['db1']['prev_nodes'], dags['db1']['modules'], appendage=dags['db1']['upsampler']))
        x = torch.cat((x,x1),axis=1)
        if verbose>=100: print('%s:%s'%(str('x'),str(x.shape)))

        x = self.conv_resize(x)

        yg = self.cyg(x)
        ys = self.cys(x)

        if verbose>=100:
            print('outputs:')
            for a,b in zip(["x", "yg", "ys"], [x,yg,ys]):
                print('%s:%s'%(str(a),str(b.shape)))
        return x, yg, ys
"""
#########################################################
Example dags
#########################################################
"""
def get_dummy_dag(mode=None, appendage='mp2'):
    if mode is None: # returns a full sample dag
        dag = {
            0: [Node(id=1,name='conv3x3')], # input
            1: [Node(id=2,name='conv3x3'),Node(id=3,name='conv3x3'),],
            2: [Node(id=4,name='fconv3x3'),Node(id=5,name='fconv3x3'),],
            3: [Node(id=6,name='avg'),],
            4: [Node(id=6,name='avg'),],
            5: [Node(id=6,name='avg'),],
            6: [Node(id=7,name=appendage),],
        }
        return dag
    elif mode == 'dag_parts' :
        prev_nodes = [0,1,1,2,2]
        modules = [ 'conv3x3', 'conv3x3', 'conv3x3', 'fconv3x3', 'fconv3x3']
        return prev_nodes, modules, appendage

def get_dummy_dag2(mode=None, appendage='mp2'):
    if mode == None:
        dag = {
            0: [Node(id=1,name='conv3x3'), Node(id=2,name='conv3x3'), 
                Node(id=5,name='fconv3x3'), Node(id=7,name='fconv3x3'), 
                Node(id=12,name='conv3x3')], # input
            1: [Node(id=3,name='conv3x3'),],
            2: [Node(id=4,name='fconv3x3'),],
            3: [Node(id=6,name='fconv3x3'), Node(id=8,name='fconv3x3')],
            4: [Node(id=9,name= 'conv3x3'), Node(id=10,name= 'fconv3x3')],
            5: [Node(id=13,name='avg'),],
            6: [Node(id=13,name='avg'),],    
            7: [Node(id=13,name='avg'),],
            8: [Node(id=13,name='avg'),],
            9: [Node(id=11,name='fconv3x3'),],
            10: [Node(id=13,name='avg'),],
            11: [Node(id=13,name='avg'),],
            12: [Node(id=13,name='avg'),],    
            13: [Node(id=14,name=appendage),],    
        }
        return dag
    elif mode == 'dag_parts':
        prev_nodes = [0,0,1,2,0,3,0,3,4,4,9,0]
        modules = [ 'conv3x3', 'conv3x3', 'conv3x3', 'fconv3x3', 'fconv3x3', 
            'fconv3x3', 'fconv3x3', 'fconv3x3', 'conv3x3', 'fconv3x3', 
            'fconv3x3', 'conv3x3']
        return prev_nodes, modules, appendage