import os, pickle
import torch 
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_shape_min_max(x):
    # x is a tensor
    print('x: %s [%s,%s]'%(str(x.shape),str(torch.min(x).item()),str(torch.max(x).item())))


def parse_bool_from_string(bool_string):
    # assume bool_string is either 0 or 1 (str)
    if str(bool_string)=='1':
        return True
    elif str(bool_string)=='0':
        return False
    else:
        raise RuntimeError('parse_bool_from_string only accepts 0 or 1.')


def manage_directories(ROOT_DIR, PROJECT_ID, model_file='net.model', verbose=100):
    if ROOT_DIR is None: 
        ROOT_DIR = os.getcwd()
    PROJECT_DIR = os.path.join(ROOT_DIR,'checkpoint',PROJECT_ID)
    MODEL_DIR = os.path.join(PROJECT_DIR,model_file)
    OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
    if not os.path.exists(PROJECT_DIR): os.mkdir(PROJECT_DIR)
    if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
    if verbose:
        print('ROOT_DIR    :', ROOT_DIR)
        print('PROJECT DIR :', PROJECT_DIR)
        print('MODEL DIR   :', MODEL_DIR)
    return ROOT_DIR, PROJECT_DIR, MODEL_DIR, OUTPUT_DIR


from .printing_manager import ShortPrint
sp = ShortPrint() 

class FastPickleClient(object):
    def __init__(self):
        super(FastPickleClient, self).__init__()
        self.save_text = 'Saving data via FastPickleClient...'
        self.load_text = 'Loading data via FastPickleClient...'
    
    def pickle_data(self, save_data, save_dir, tv=(0,0,None), text=None):
        if text is not None: 
            self.save_text = text
        output = open(save_dir, 'wb')
        pickle.dump(save_data, output)
        output.close()      
        sp.prints('%s\n  %s'%(str(self.save_text),str(save_dir)), tv=tv)

    def load_pickled_data(self, pickled_dir, tv=(0,0,None), text=None):
        if text is not None:
            self.load_text = text
        pkl_file = open(pickled_dir, 'rb')
        this_data = pickle.load(pkl_file)
        pkl_file.close()        
        sp.prints('%s\n  %s'%(str(self.load_text),str(pickled_dir)), tv=tv)
        return this_data

from skimage.transform import resize
def numpy_batch_CHW_resize(x, target_shape, is_pytorch_tensor=False):
    # expected input shape is (batch, C, H, W)
    # compatible with pytorch tensors' shape
    # target shape is C, H, W
    if is_pytorch_tensor:
        x = x.clone().detach().cpu().numpy()
    
    s = target_shape 
    batch_size= x.shape[0]
    C, H, W = target_shape
    assert (x.shape[1]==C)

    output = np.zeros(shape=(batch_size,C,H,W))
    for i in range(batch_size):
        temp = x[i].transpose(1,2,0)
        temp = resize(temp, (H,W,C))
        temp = temp.transpose(2,0,1)
        output[i] = temp
    if is_pytorch_tensor:
        output = torch.tensor(output, requires_grad=False) 
    return output

def average_every_n(x,iter_list=None, n=10):
    """
    N_iter = 1223
    # iter_list = None 
    iter_list = 777 + np.array(range(N_iter)) # also try this
    y = np.exp(-np.linspace(0,5,N_iter)) + 0.1*np.random.randn(N_iter)
    x1, y1 = average_every_n(y, iter_list=iter_list,n=100)

    plt.figure(figsize=(4,6))
    plt.gcf().add_subplot(211)
    if iter_list is None:
        plt.plot(y,)
    else:
        plt.plot(iter_list,y,)
    plt.gcf().add_subplot(212)
    plt.plot(x1,y1,)

    plt.show()
    """
    n_excess = len(x)%n
    n_av = len(x)//n

    last_round_index = int(n_av*n)
    if iter_list is None:
        iter_list = np.array(range(1,1+len(x)))
    else:
        assert(len(iter_list)==len(x))
    x1 = np.array(iter_list[:last_round_index]).reshape(n_av,n)
    y1 = np.array(x[:last_round_index]).reshape(n_av,n)

    x1 = x1[:,-1]
    y1 = np.mean(y1,axis=1)
    x1 = list(x1)
    y1 = list(y1)

    if n_excess>0:
        x1.append(np.mean(iter_list[last_round_index:]))
        y1.append(x[-1])
    return x1, y1 


