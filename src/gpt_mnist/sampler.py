import torch
import numpy as np
import src.gpt_mnist.data as ud
from .data import SimpleGen
from scipy.ndimage import rotate

class GPT_MNIST_Sampler():
    def __init__(self, compenv_mode=None, growth_mode=None,
        verbose=0):
        super(GPT_MNIST_Sampler, self).__init__()

        self.L =  28
        self.G0 = np.array([[0,0,0,0,0,0,0,0], # 1
                       [2,3,4,0,0,0,0,0], 
                       [0,0,0,0,0,0,0,2],
                       [0,0,0,0,0,0,0,0], # terminal
                      ])
        self.topology_matrix = np.array([[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]]) 

        self.gen = SimpleGen(self.L, G0=self.G0, topology_matrix=self.topology_matrix, 
            compenv_mode=compenv_mode, growth_mode=growth_mode, verbose=verbose)

        self.compenv_mode = compenv_mode
        self.growth_mode = growth_mode
        
    def get_one_sample(self, c):
        # c = 0,1,...,9, the class
        # nBSG = self.gen.nBSG
        if c==0:
            img, seeds, params = self.get_class_type001(0)
        elif c==1:
            img, seeds, params = self.get_class_type001(1)
        elif c==2:
            img, seeds, params = self.get_class_type001(2)
        elif c==3:
            img, seeds, params = self.get_class_type001(3)
        elif c==4:
            img, seeds, params = self.get_class_type001(4)
        elif c==5:
            img, seeds, params = self.get_class_type001(5)
        elif c==6:
            img, seeds, params = self.get_class_type001(6)
        elif c==7:
            img, seeds, params = self.get_class_type001(7)
        elif c==8:
            img, seeds, params = self.get_class_type001(8)
        elif c==9:
            img, seeds, params = self.get_class_type001(9)
        return img, seeds, params # they will be x,y later for NN training


    def get_class_type001(self, c):
        """
        return
            img
              Image obtained after applying growth function on seeds.
            seeds
              Configuration matrix. Each element is 1-based index of GE.
            params
              Any other settings related to the configuration.
        """
        board_shape = (self.L,self.L)
        seeds = np.ones(shape=board_shape) 
        board = np.ones(shape=board_shape)
        
        deg = np.random.uniform(-10,10)
        config_param = {
            'T_MAX':np.random.randint(8,12),
            'x_translation': np.random.randint(-1,1+1),
            'y_translation': np.random.randint(-2,2+1),
            'rotate': deg, # deg
        }
        g_params = [
            {}, # only 1 gen
        ]

        params = (config_param, g_params)

        tx, ty = config_param['x_translation'], config_param['y_translation']
        
        if c==0: 
            x,y = 7+tx,14+ty
            seeds[y,x] = ud.code_generator(2, 2+1, self.gen.nBSG) # base. Generator 2 rotated twice w.r.t topology_matrix
        elif c==1: 
            x,y = 14+tx,7+ty
            seeds[y,x] = ud.code_generator(2, 4+1, self.gen.nBSG) 
        elif c==2:
            config_param['T_MAX'] +=1
            x,y = 8+tx,14+ty
            seeds[y,x] = ud.code_generator(2, 1+1, self.gen.nBSG) 
            x,y = 13+tx,8+ty
            seeds[y,x] = ud.code_generator(2, 3+1, self.gen.nBSG) 
        elif c==3:
            x,y = 8+tx,18+ty
            seeds[y-2,x] = ud.code_generator(2, 1+1, self.gen.nBSG) 
            seeds[y-config_param['T_MAX'] ,x+config_param['T_MAX']] = ud.code_generator(2, 1+4, self.gen.nBSG) 
            seeds[y-1,x] = ud.code_generator(2, 2+1, self.gen.nBSG)  
        elif c==4:
            x,y = 8+tx,8+ty
            seeds[y,x] = ud.code_generator(2, 2+1, self.gen.nBSG) 
            seeds[y+8,x+2] = ud.code_generator(2, 1, self.gen.nBSG) 
            seeds[y+2,x+6+np.random.randint(0,3)] = ud.code_generator(2, 1+4, self.gen.nBSG) 
        elif c==5:
            x,y = 8+tx,10+ty
            p = int(np.random.randint(6,10))
            seeds[y,x+3] = ud.code_generator(2, 2+1, self.gen.nBSG)
            seeds[y+p,x] = ud.code_generator(2, 2+1, self.gen.nBSG) 
            seeds[y+p+1,x+int(0.5*config_param['T_MAX'])] = ud.code_generator(2, 1, self.gen.nBSG) # base
        elif c==6:
            x,y = 7+tx,6+ty
            p = np.random.randint(0,3)
            seeds[y,x] = ud.code_generator(2, 2+1, self.gen.nBSG) 
            seeds[y,x+5 +p] = ud.code_generator(2, 3+1, self.gen.nBSG) 
            seeds[y+ config_param['T_MAX'] , x+6 +p] = ud.code_generator(2, 1+1, self.gen.nBSG) 
            seeds[y+8,x+p] = ud.code_generator(2, 1, self.gen.nBSG) 
        elif c==7:
            x,y = 3+tx,6+ty
            config_param['T_MAX'] -= 2
            p = np.random.randint(0,3)
            seeds[y,x] = ud.code_generator(2, 2+1, self.gen.nBSG) 
            seeds[y,x+5+p] = ud.code_generator(2, 3+1, self.gen.nBSG) 
            seeds[y,x+5+p + np.random.randint(7,9)] = ud.code_generator(2, 4+1, self.gen.nBSG) 
            seeds[y+8,x+p] = ud.code_generator(2, 1, self.gen.nBSG) 
        elif c==8:
            x,y = 7+tx,8+ty
            p = np.random.randint(0,3)
            seeds[y,x+2] = ud.code_generator(2, 2+1, self.gen.nBSG) 
            seeds[y+11+p,x+9] = ud.code_generator(2, 6+1, self.gen.nBSG) 
            seeds[y+6,x+7] = ud.code_generator(2, 7+1, self.gen.nBSG) 
            seeds[y+11+p,x] = ud.code_generator(2, 1+1, self.gen.nBSG) 
        elif c==9:
            config_param['T_MAX'] = 8
            x,y = 7+tx,14+ty
            seeds[y+np.random.randint(0,2),1+x] = ud.code_generator(2, 2+1, self.gen.nBSG) 
            seeds[y+1,x] = ud.code_generator(2, 3+1, self.gen.nBSG) 
            seeds[y-1,x] = ud.code_generator(2, 1+1, self.gen.nBSG) 
            seeds[y,x+config_param['T_MAX']+2] = ud.code_generator(2, 5+1, self.gen.nBSG) 
            seeds[y-1,x+config_param['T_MAX']] = ud.code_generator(2, 7+1, self.gen.nBSG) 
        
        ##############################
        # main GPT-based mechanism here
        CHANGE = seeds > 0
        board = board *(1-CHANGE )+ seeds* CHANGE
        board = self.gen.develop(config_param['T_MAX'], board, compenv_mode=self.compenv_mode, growth_mode=self.growth_mode)
        ##############################

        img = ((board-1)>0.).astype(float)
        if np.abs(deg)<2:
            img = rotate(rotate(img,5,reshape=False),-5,reshape=False)
        img = rotate(img,config_param['rotate'],reshape=False)

        return img, seeds, params

class Pytorch_GPT_MNIST_Sampler(GPT_MNIST_Sampler):
    def __init__(self, compenv_mode=None, growth_mode=None,
        verbose=0):
        super(Pytorch_GPT_MNIST_Sampler, self).__init__()
                
    def get_sample_batch_uniform_random(self, batch_size=16, device=None):
        class_indices = np.random.choice(range(10),size=batch_size)
        img_batch,y_batch,yg_batch, ys_batch = self.get_sample_batch(class_indices, device=device)
        return img_batch,y_batch,yg_batch, ys_batch

    def get_sample_batch(self, class_indices, device=None):
        img_batch = []
        y_batch, yg_batch, ys_batch = [], [], []
        for c in class_indices:
            img, seeds, params = self.get_one_sample(c)
            img_batch.append([img])

            # yg  : generators' type (in 1-based indexing, it is alpha+1) and position matrix
            # ys : 1-based inxed of element of BSG, used to transform elements of y
            yg, ys = ud.decode_generator(seeds, self.gen.nBSG) 
            yg_batch.append(yg)
            ys_batch.append(ys)
            y_batch.append(c)

        img_batch = torch.tensor(img_batch).to(torch.float)
        y_batch = torch.tensor(y_batch).to(torch.long)
        yg_batch = torch.tensor(yg_batch).to(torch.long) - 1 # return it to 0-based index
        ys_batch = torch.tensor(ys_batch).to(torch.long) - 1

        if not device is None:
            img_batch, y_batch, yg_batch, ys_batch = img_batch.to(device),y_batch.to(device),yg_batch.to(device), ys_batch.to(device)
        
        return img_batch,y_batch,yg_batch, ys_batch