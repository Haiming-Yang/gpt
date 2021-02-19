import torch
import torch.nn as nn
import collections

Node = collections.namedtuple('Node', ['id', 'name'])

class Builder(nn.Module):
    def __init__(self, ):
        super(Builder, self).__init__()
        self.names_to_cmodules = {
            'avg': lambda x: x, # do nothing. Average only at the end.
        }

    def forward(self):
        raise RuntimeError('Only to be implemented in the child class.')

    def compute_through_dag(self, x, num_blocks, dag, verbose=0):
        C = self.names_to_cmodules

        output = None
        y = {0:x}
        n_leaf = 0
        for i in range(1+num_blocks):
            if verbose>=100: print('\nPROCESSING NODE: ', i)
            for node in dag[i]:
                this_id = node.id
                mod_name = node.name
                if verbose>=100: print(node)
                if i==0:
                    mod_name = mod_name+ '_init'

                node_value = C[mod_name](y[i])

                if this_id<=num_blocks:
                    y[this_id] = node_value
                    if verbose>=100: print('storing node:',this_id)
                else:
                    n_leaf += 1
                    if output is None:    
                        output = node_value
                    else:
                        output += node_value
                    if verbose>=100: print('output for averaging')
        if verbose>=100: print('\nn_leaf=', n_leaf)
        output = output/n_leaf
        return output

    def construct_dag(self, prev_nodes, modules, appendage=None):
        """
        See examples input and output in get_dummy_dag()
        """
        n_external_node = 1 if appendage is None else 2
        dag = collections.defaultdict(list)

        for this_node, (prev_node, mod) in enumerate(zip(prev_nodes, modules)):
            dag[prev_node].append(Node(id=this_node+1,name=mod))

        leaf_nodes = set(range(self.num_blocks+1)) - dag.keys()
        for i in leaf_nodes:
            dag[i].append(Node(id=self.num_blocks+1,name='avg'))

        if appendage is not None:
            appendage_i = self.num_blocks + 1 + 1
            dag[appendage_i-1].append(Node(id=appendage_i,name=appendage))

        assert(self.num_blocks == len(dag)-n_external_node)
        # raise Exception('gg')
        return dag

    def get_vanilla_conv(self, input_channel):
        return nn.Sequential(
            nn.Conv2d(input_channel,64,3,padding=1, padding_mode='replicate', bias=False),
            nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,padding=1, padding_mode='replicate', bias=False), 
            nn.ReLU(),nn.BatchNorm2d(64),)

    def get_one_funnel_conv(self, input_channel, act='ReLU'):
        if act =='ReLU':
            act = nn.ReLU()

        seq =  nn.Sequential(
            nn.Conv2d(input_channel,64, 3, bias=False),
            act, nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64,64, 3, bias=False),
            act, nn.BatchNorm2d(64),
            )
        return seq        

        

class BENASConvBlock(Builder):
    def __init__(self, input_channel=64, num_blocks=5):
        super(BENASConvBlock, self).__init__()

        self.num_blocks = num_blocks

        self.vanilla_conv_init = self.get_vanilla_conv(input_channel)
        self.vanilla_conv = self.get_vanilla_conv(64)
        self.one_funnel_conv_init = self.get_one_funnel_conv(input_channel, act='ReLU')
        self.one_funnel_conv = self.get_one_funnel_conv(64, act='ReLU')

        self.names_to_downsampler = {
            'mp2' : nn.MaxPool2d(2),
        }

        self.names_to_cmodules = {
            'conv3x3_init' : self.vanilla_conv_init,
            'conv3x3' : self.vanilla_conv,
            'fconv3x3_init' : self.one_funnel_conv_init,        
            'fconv3x3' : self.one_funnel_conv, 
            'avg': lambda x: x, # do nothing. Average only at the end.
        }


    def forward(self, x, num_blocks, dag, verbose=0):
        # num_blocks: exclude the downsampler
        output = self.compute_through_dag(x, num_blocks, dag, verbose=verbose)

        if 1+num_blocks in dag:
            if verbose>=100: print('\nDownsampling...')
            D = self.names_to_downsampler
            node = dag[1+num_blocks][0]
            mod_name = node.name
            output_downsampled = D[mod_name](output)
            return output, output_downsampled
    
        return output, _


class BENASDeConvBlock(Builder):
    def __init__(self, input_channel=64, num_blocks=5):
        super(BENASDeConvBlock, self).__init__()
        self.num_blocks = num_blocks

        self.vanilla_conv_init = self.get_vanilla_conv(input_channel)
        self.vanilla_conv = self.get_vanilla_conv(64)
        self.one_funnel_conv_init = self.get_one_funnel_conv(input_channel, act='ReLU')
        self.one_funnel_conv = self.get_one_funnel_conv(64, act='ReLU')

        self.names_to_upsampler = {
            'us2' : nn.Upsample(scale_factor=2),
        }

        self.names_to_cmodules = {
            'conv3x3_init' : self.vanilla_conv_init,
            'conv3x3' : self.vanilla_conv,
            'fconv3x3_init' : self.one_funnel_conv_init,        
            'fconv3x3' : self.one_funnel_conv, 
            'avg': lambda x: x, # do nothing. Average only at the end.
        }

    def forward(self, x, num_blocks, dag, verbose=0):
        # num_blocks: exclude the upsampler
        output = self.compute_through_dag(x, num_blocks, dag, verbose=verbose)

        if 1+num_blocks in dag:
            if verbose>=100: print('\nUpsampling...')
            D = self.names_to_upsampler
            node = dag[1+num_blocks][0]
            mod_name = node.name
            output_upsamples = D[mod_name](output)
            return output, output_upsamples

        return output, _
