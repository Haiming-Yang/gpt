import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision

import os, time, PIL
import numpy as np
import matplotlib.pyplot as plt

from . import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def entry(args):
    print('pipeline.entry() mode: %s'%(str(args['mode'])))

    if args['mode'] == 'training':
        training(args)
    elif args['mode'] == 'sample':
        sample(args)
    elif args['mode'] == 'plot_results':
        plot_results(args)
    else:
        print('invalid --mode argument? Use --h to check available modes.')

def sample(args):
    print('sample()')
    rcon = Reconstructor()
    rcon.sample(args)

def training(args):
    print('training()')

    tr = Trainer()
    args = tr.run_training(args)
    print_settings(args)

class Trainer(utils.FastPickleClient):
    def __init__(self, ):
        super(Trainer, self).__init__()
    
    # inherited from FastPickleClient
    # def pickle_data(self, save_data, save_dir, tv=(0,0,None), text=None):
    # def load_pickled_data(self, pickled_dir, tv=(0,0,None), text=None):

    def run_training(self, args):
        print('Trainer.run_training() on %s dataset'%(str(args['dataset'])))

        OBSERVE = utils.parse_bool_from_string(args['OBSERVE'])
        EVALUATE = utils.parse_bool_from_string(args['EVALUATE'])
        DO_DEBUG = utils.parse_bool_from_string(args['debug_mode'])
        REALTIME_PRINT = utils.parse_bool_from_string(args['realtime_print'])

        N_EPOCH = args['N_EPOCH']
        PROJECT_ID = args['PROJECT_ID']
        batch_size = args['batch_size'] 
        LEARNING_RATE = args['learning_rate']
        DATA_DIR = 'data' if not 'DATA_DIR' in args else args['DATA_DIR']
        ROOT_DIR, PROJECT_DIR, MODEL_DIR, OUTPUT_DIR = utils.manage_directories(os.getcwd(), PROJECT_ID)
        AVG_LOSS_EVERY_N_ITER = args['AVG_LOSS_EVERY_N_ITER']
        RESULT_DATA_DIR = OUTPUT_DIR + '.data'
        BEST_MODEL_DIR = MODEL_DIR + '.best'
        SAVE_IMG_EVERY_N_ITER = args['SAVE_IMG_EVERY_N_ITER']
        LOSS_IMG_DIR = OUTPUT_DIR + '.loss.jpg'
        if DO_DEBUG:
            args['N_EPOCH'] = 2
            args['n'] = 24
            STOP_AT_N_ITER = args['STOP_AT_N_ITER']

        # loading data
        trainloader, evalloader = self.get_data_loader(args)
        evaliter = iter(evalloader)
        IMG_SHAPE, is_black_and_white = self.get_image_shape(args)    
        output_data_dict = self.get_result_dictionary(RESULT_DATA_DIR)          

        # prepare model
        if os.path.exists(BEST_MODEL_DIR): net = self.init_or_load_model(BEST_MODEL_DIR, args)
        else: net = self.init_or_load_model(MODEL_DIR, args)
        net.to(device=device)
        print('n params:', utils.count_parameters(net))

        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, betas=(0.5,0.999), weight_decay=0)

        rcon = Reconstructor()

        if EVALUATE:
            print('preparing for FID with n:%s'%(str(args['n'])))
            prepare_data_for_FID(args)

        start = time.time()
        print('\nStart training...')
        for epoch in range(N_EPOCH):
            len_data = len(trainloader)

            for i, data in enumerate(trainloader):
                net.total_iter+=1
                net.train()
                net.zero_grad()

                x, labels = data
                x = x.to(torch.float).to(device=device) 

                # tensor_batch_to_display(x,raise_exception=True)
                
                y = net(x)
                total_loss = criterion(x,y)
                loss = total_loss
                realtime_print_options = {}

                total_loss.backward()        
                optimizer.step()

                # ignore the fact that it may slow down the process a fair bit in the initial stage.
                output_data_dict['iters'].append(net.total_iter)
                output_data_dict['losses'].append(loss.item())
                if net.total_iter > AVG_LOSS_EVERY_N_ITER and net.total_iter>100:
                    running_avg_loss = np.mean(output_data_dict['losses'][-AVG_LOSS_EVERY_N_ITER:])
                    if net.best_avg_loss > running_avg_loss:
                        net.best_avg_loss = running_avg_loss
                        avg_loss_iter, avg_loss = utils.average_every_n(output_data_dict['losses'],iter_list=output_data_dict['iters'], n=AVG_LOSS_EVERY_N_ITER)
                        
                        torch.save(net, BEST_MODEL_DIR)
                        save_loss_image(avg_loss_iter, avg_loss, dir=LOSS_IMG_DIR)
                        self.pickle_data(output_data_dict, RESULT_DATA_DIR, tv=(0,0,100), text=None)
                        save_reconstructed_images(x, net, OUTPUT_DIR, model=args['model'], IMG_SHAPE=IMG_SHAPE, best_loss_recon=True)


                if SAVE_IMG_EVERY_N_ITER>0:
                    if (i+1)%SAVE_IMG_EVERY_N_ITER==0 or (i+1)==len_data:
                        save_reconstructed_images(x, net, OUTPUT_DIR,model=args['model'], IMG_SHAPE=IMG_SHAPE, )
                        save_reconstructed_images(x, net, OUTPUT_DIR,model=args['model'], IMG_SHAPE=IMG_SHAPE, difference=True)

                if REALTIME_PRINT:
                    if (i+1)%4==0 or (i+1)>=len_data: 
                        update_str = self.make_update_text(epoch, N_EPOCH, i, len_data, net.best_avg_loss,**realtime_print_options)
                        print('%-96s'%(str(update_str)),end='\r')

                if DO_DEBUG:
                    if (i+1)%STOP_AT_N_ITER==0: break

            x, _ = next(evaliter)
            x = x.to(torch.float).to(device=device)
            save_reconstructed_images(x, net, OUTPUT_DIR,model=args['model'], IMG_SHAPE=IMG_SHAPE, black_and_white=is_black_and_white)
            
            torch.save(net, MODEL_DIR)
            
            if EVALUATE:
                rcon.reconstruct_images_into_folder(args, do_compute_SSIM=True, realtime_print=REALTIME_PRINT)
                fid_value = rcon.compute_fid(args, net)

        print('\ntraining ended at net.total_iter=%s.'%(str(int(net.total_iter))))

        end = time.time()
        elapsed = end - start
        print(' time taken %s[s] = %s [min] = %s [hr]'%(str(round(elapsed,1)), str(round(elapsed/60.,1)), str(round(elapsed/3600.,1))))
        return args


    def make_update_text(self, epoch, N_EPOCH, i, len_data, best_avg_loss, **kwargs):
        update_str = 'epoch:%s/%s iter: %s/%s best_avg_recon_loss:%s'%(str(epoch+1),str(N_EPOCH),str(i+1),
            str(len_data),str(np.round(best_avg_loss,5)))

        for item_name, update_items in kwargs.items():
            update_val = update_items['value']
            if update_items['rounding'] is not None:
                update_val = round(update_val,update_items['rounding'])
            update_str += ' %s:%s'%(str(item_name),str(update_val))
        return update_str

    def get_data_loader(self, args):
        DATA_DIR = args['DATA_DIR']
        batch_size = args['batch_size']
        if  args['dataset'] == 'cifar10':
            from .data import prepare_cifarloader
            trainloader = prepare_cifarloader(train=True, root_dir=DATA_DIR, batch_size=batch_size, shuffle=True, demo=0, download=0)
            evalloader = prepare_cifarloader(train=False, root_dir=DATA_DIR, batch_size=16, shuffle=True, demo=0, download=0)
            return trainloader, evalloader         
        elif  args['dataset'] == 'celeba64':
            from .data import prepare_celebaloader
            # Put celeba/img_align_celeba.zip folder in the "data" folder if you set DATA_DIR to 'data' 
            if DATA_DIR == 'data':
                DATA_DIR = os.path.join(DATA_DIR, 'celeba', 'img_align_celeba.zip')
            trainloader = prepare_celebaloader(img_size=(64,64),train=True, root_dir=DATA_DIR, batch_size=batch_size, shuffle=True)
            evalloader = prepare_celebaloader(img_size=(64,64),train=False, root_dir=DATA_DIR, batch_size=16, shuffle=True)
            return trainloader, evalloader  
        else:
            raise RuntimeError('Invalid dataset?')

    def get_result_dictionary(self, RESULT_DATA_DIR):
        if os.path.exists(RESULT_DATA_DIR): 
            output_data_dict = self.load_pickled_data(RESULT_DATA_DIR, tv=(0,0,None), text=None)
        else: 
            output_data_dict = {'iters':[],'losses':[]}
        return output_data_dict

    def init_or_load_model(self, MODEL_DIR, args):
        OBSERVE = utils.parse_bool_from_string(args['OBSERVE'])
        if os.path.exists(MODEL_DIR):
            net = torch.load(MODEL_DIR)
            print('loading model...%s at %s iter'%(str(type(net)),str(net.total_iter)))
            net.OBSERVE = OBSERVE
        else:
            if args['dataset'] == 'cifar10' and args['model']=='SimpleGrowth':
                from .model import SimpleGrowth
                net = SimpleGrowth(img_shape=(3,32,32), hidden_layer=3, OBSERVE=OBSERVE)
            elif args['dataset'] == 'celeba64' and args['model']=='SimpleGrowth':
                from .model import SimpleGrowth
                net = SimpleGrowth(img_shape=(3,64,64), hidden_layer=4, OBSERVE=OBSERVE)
            else:
                raise RuntimeError('Model not found. Check --dataset or --model')   
            print('initiating new model...%s'%(str(type(net))))
        return net

    def get_image_shape(self, args):
        if  args['dataset'] == 'cifar10':
            IMG_SHAPE = (3,32,32) # (C,H,W)
            is_black_and_white=False
        elif args['dataset'] == 'celeba64':
            IMG_SHAPE = (3,64,64) # (C,H,W)
            is_black_and_white=False            
        else:
            raise RuntimeError('Invalid dataset?')
        return IMG_SHAPE, is_black_and_white


#######################
# Some utils
#######################

def save_reconstructed_images(x, net, OUTPUT_DIR,model='SimpleGrowth', IMG_SHAPE=(1,28,28), black_and_white=False, difference=False, best_loss_recon=False):
    by_batch = True
    if IMG_SHAPE == (3,218,178):
        by_batch = False
    C, H, W = IMG_SHAPE
    net.eval()


    if by_batch:
        y = net(x)
        
        y = y.clone().detach().cpu().numpy()
        x = x.clone().detach().cpu().numpy()

        if difference:
            y = np.clip((y-x)**2,0,1.)**0.5

        n_batch = y.shape[0]

        # print(n_batch)
        if n_batch>16: 
            y_reconstruct = y[:16]
            x_compare = x[:16]
        else:
            x_compare = np.zeros(shape=(16,C,H,W))
            y_reconstruct = np.zeros(shape=(16,C,H,W))
            y_reconstruct[:n_batch] = y
            x_compare[:n_batch] = x
    else:
        x_compare = np.zeros(shape=(16,C,H,W))
        y_reconstruct = np.zeros(shape=(16,C,H,W))        
        batch_size = x.shape[0]
        for i in range(batch_size):
            y = net(x[i:i+1])
            y = y.clone().detach().cpu().numpy()
            x1 = x[i:i+1].clone().detach().cpu().numpy()
            y_reconstruct[i] = y
            x_compare[i] = x1                        
            if i>=16: break

    plt.figure(figsize=(5,10))
    for i in range(16):
        plt.gcf().add_subplot(8,4,i+1)
        if black_and_white:
            plt.gca().imshow(y_reconstruct[i][0], vmin=0,vmax=1, cmap='gray')
        else:
            plt.gca().imshow(y_reconstruct[i].transpose(1,2,0))
        set_figure_setting(i, n_last=16, set_title='reconstructed')

    for i in range(16):
        plt.gcf().add_subplot(8,4,16 + i+1)
        if black_and_white:
            plt.gca().imshow(x_compare[i][0], vmin=0,vmax=1, cmap='gray')
        else:
            plt.gca().imshow(x_compare[i].transpose(1,2,0))
        set_figure_setting(i, n_last=16, set_title='original')

    plt.tight_layout()
    if best_loss_recon:
        img_name = 'recons_best.jpeg'
    else:
        if not difference:
            img_name = 'recons_%s.jpeg'%(str(1000000+net.total_iter))[1:]
        else:
            img_name = 'recons_%s_diff.jpeg'%(str(1000000+net.total_iter))[1:]

    IMG_DIR = os.path.join(OUTPUT_DIR, img_name)
    plt.savefig(IMG_DIR)
    plt.close()

def set_figure_setting(i, n_last, set_title=None):
    if i==0:
        plt.gca().set_xticks([])
        if set_title:
            plt.gca().set_title(set_title)
    elif i+1==n_last:
        plt.gca().set_yticks([])
    else:
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])


def manage_dir_for_FID(dataset_name, ROOT_DIR):
    if ROOT_DIR is None: ROOT_DIR = os.getcwd()
    ckpt_dir = os.path.join(ROOT_DIR,'checkpoint')
    fid_folder_dir = os.path.join(ckpt_dir,'for_fid') 
    fid_data_dir = os.path.join(fid_folder_dir,dataset_name)

    for x in [ckpt_dir, fid_folder_dir, fid_data_dir]:
        if not os.path.exists(x): 
            os.mkdir(x)
    return fid_data_dir


def manage_directories2(args):
    PROJECT_ID = args['PROJECT_ID']
    
    ROOT_DIR, PROJECT_DIR, MODEL_DIR, OUTPUT_DIR = utils.manage_directories(os.getcwd(), PROJECT_ID, verbose=0)
    RECON_DIR = os.path.join(PROJECT_DIR,'imgs_for_FID')
    RESULT_DIR = os.path.join(PROJECT_DIR,'recons.result')
    if not os.path.exists(RECON_DIR): os.mkdir(RECON_DIR)
    return ROOT_DIR, PROJECT_DIR, MODEL_DIR, OUTPUT_DIR, RECON_DIR, RESULT_DIR  

def print_settings(args):
    print('='*64)
    for x, y in args.items():
        print('%s:%s'%(str(x),str(y)))

def save_loss_image(avg_loss_iter, avg_loss, dir):
    plt.figure()
    plt.plot(avg_loss_iter, avg_loss)
    plt.savefig(dir)
    plt.close()

##################################
# Prep for evaluation
##################################

def prepare_data_for_FID(args):
    print('prepare_data_for_FID()')

    ROOT_DIR = os.getcwd()
    FID_DATA_DIR = manage_dir_for_FID(args['dataset'], ROOT_DIR)
    n_data = args['n'] if args['n']>8 else 8
    print('preparing FID comparison dataset for %s'%(str(args['dataset'])))
    print('FID_DATA_DIR:%s'%(str(FID_DATA_DIR)))
    print('n_data: %s'%(str(n_data)))

    if args['dataset'] == 'cifar10':
        from .data import CIFAR10Dataset
        ds = CIFAR10Dataset(train=False, root_dir=args['DATA_DIR'],download=False)
        n_channel = 3
    elif args['dataset'] == 'celeba64':
        from .data import CelebADataset
        DATA_DIR = args['DATA_DIR']
        if DATA_DIR == 'data':
            DATA_DIR = os.path.join(DATA_DIR, 'celeba', 'img_align_celeba.zip')
        ds = CelebADataset(DATA_DIR, img_size=(64,64))
        n_channel = 3
    else:
        raise RuntimeError('Invalid --dataset?')

    import random
    n_available = ds.__len__()
    indices = np.array(range(n_available))
    random.shuffle(indices)

    if n_data>n_available: n_data = n_available

    import PIL 
    for i in range(n_data):
        x,y0 = ds.__getitem__(indices[i]) # min, max range is [0,1]
        if isinstance(x, type(torch.tensor([1.]) )):
            x = x.clone().detach().numpy()
        x = (x.transpose(1,2,0) * 255.).astype(np.uint8)
        
        filename = os.path.join(FID_DATA_DIR, '%s.%s.jpg'%(str(args['dataset']),str(i)))
        if n_channel==3:
            img = PIL.Image.fromarray(x)
        elif n_channel==1:
            x = np.concatenate((x,x,x),axis=2)
            img = PIL.Image.fromarray(x)
        else:
            raise RuntimeError('wrong channel number?!')
        
        img.save(filename)
        # print(x.shape, y0, '[%s,%s]'%(str(np.min(x)),str(np.max(x))))    


class Reconstructor(Trainer):
    def __init__(self):
        super(Reconstructor, self).__init__()

    def sample(self, args):
        DATA_DIR = args['DATA_DIR']
        ROOT_DIR, PROJECT_DIR, MODEL_DIR, OUTPUT_DIR, RECON_DIR, RESULT_DIR = manage_directories2(args)

        net = self.init_or_load_model(MODEL_DIR, args)
        net.to(device=device)
        net.eval()

        if args['model']=='SimpleGrowth'  and args['dataset']=='cifar10':
            latent_dim = 15
            latent_img_shape = (4,4)
        elif args['model']=='SimpleGrowth'  and args['dataset']=='celeba64':
            latent_dim = 15
            latent_img_shape = (4,4)
        else:
            raise RuntimeError('Invalide dataset or model?')

        img_folder = os.path.join(OUTPUT_DIR,'sample')
        if not os.path.exists(img_folder): os.mkdir(img_folder)

        sampling_mode = args['sampling_mode']
        img_samples, img_dir = self.make_img_samples(net, latent_dim, latent_img_shape, img_folder, args)
        
        save_sixteen_images(img_samples, img_dir, title=None)


    def make_img_samples(self, net, latent_dim , latent_img_shape, img_folder, args):
        batch_size = 16
        mode=  args['sampling_mode']
        H, W = latent_img_shape
        if mode is None:
            feature_vec = 2*torch.rand(size=(batch_size, latent_dim*H*W))-1
            feature_vec = feature_vec.to(torch.float).to(device=device)
            # print(feature_vec.shape)
            # raise Exception('gg')
            img_samples = net.sample( (batch_size,latent_dim,H,W), latent_vec=feature_vec).clone().detach().cpu().numpy().transpose(0,2,3,1)
            img_dir = os.path.join(img_folder,'samples.jpeg')
        elif mode == 'transit':
            args['batch_size'] = 2
            trainloader, _ = self.get_data_loader(args)

            i,data = next(enumerate(trainloader))
            x, _  = data
            x = x.to(torch.float).to(device=device) 

            b, C, H0, W0 = x.shape
            latents = net.en(net.conv1(x)).reshape(b,-1)

            _,ldim = latents.shape
            diff = (latents[1]-latents[0])/15
            z = torch.zeros(size=(16,ldim))
            for i in range(16):
                z[i] = latents[0] + diff *i
            z = z.to(torch.float).to(device=device) 
            img_samples = net.sample( (batch_size,latent_dim,H,W), latent_vec=z).clone().detach().cpu().numpy().transpose(0,2,3,1)
            img_dir = os.path.join(img_folder,'samples_transit.jpeg')
        return img_samples, img_dir
    
    def compute_fid(self, args, model):
        print('compute_fid()')
        ROOT_DIR, PROJECT_DIR, _ , OUTPUT_DIR, RECON_DIR, RESULT_DIR = manage_directories2(args)

        from .pipeline import manage_dir_for_FID
        FID_DATA_DIR = manage_dir_for_FID(args['dataset'], ROOT_DIR)

        if not os.path.exists(RESULT_DIR):
            RESULT_DICT = {}
        else:
            TAB_LEVEL = 1
            RESULT_DICT = self.load_pickled_data(RESULT_DIR, tv=(TAB_LEVEL,0,100), text=None)

        print('  Comparing these two folders:',)
        print('  %s\n  %s'%(str(FID_DATA_DIR),str(RECON_DIR)))

        batch_size=1
        from .evaluation_modules.pytorch_fid.fid_score import calculate_fid_given_paths
        paths = [FID_DATA_DIR,RECON_DIR]
        fid_value = calculate_fid_given_paths(paths, batch_size, device, dims=2048)
        
        RESULT_DICT[int(model.total_iter)]['fid_score'] = fid_value
        print_result_dict(RESULT_DICT, this_iter=model.total_iter)
        
        TAB_LEVEL = 1
        self.pickle_data(RESULT_DICT, RESULT_DIR, tv=(TAB_LEVEL,0,100), text=None)

        return fid_value

        
    def reconstruct_images_into_folder(self, args, do_compute_SSIM=True, realtime_print=False):
        """
        This reconstruct images for FID computation.
        Along the way, if do_compute_SSIM is True, then we also compute the SSIM and MS SSIM values.
        """
        print('\nreconstruct_images_into_folder()')
        DATA_DIR = args['DATA_DIR']
        ROOT_DIR, PROJECT_DIR, MODEL_DIR, OUTPUT_DIR, RECON_DIR, RESULT_DIR = manage_directories2(args)

        N_DATA = 8 if args['n']<=8 else args['n']

        if args['dataset'] == 'cifar10':
            from .data import prepare_cifarloader 
            evalloader = prepare_cifarloader(train=False, root_dir=DATA_DIR, batch_size=1, shuffle=True, demo=0, download=0)
        elif args['dataset'] == 'celeba64':
            from .data import prepare_celebaloader
            if DATA_DIR == 'data':
                DATA_DIR = os.path.join(DATA_DIR, 'celeba', 'img_align_celeba.zip')
            evalloader = prepare_celebaloader(img_size=(64,64),train=False, root_dir=DATA_DIR, batch_size=1, shuffle=True)
        else:
            raise RuntimeError('invalid --dataset option?')

        net = self.init_or_load_model(MODEL_DIR, args)
        net.to(device=device)
        net.eval()

        if do_compute_SSIM:
            if not os.path.exists(RESULT_DIR):
                print('  Compute SSIM? YES. Creating new result dictionary...')
                RESULT_DICT = {}
            else:
                print('  Compute SSIM? YES. Loading result dictonary...')
                TAB_LEVEL = 1
                RESULT_DICT = self.load_pickled_data(RESULT_DIR, tv=(TAB_LEVEL,0,100), text=None)
            ORIGINAL_IMG_ARRAY, RECONS_IMG_ARRAY = [], []
            c = 0 # counters. We want to reduce memory usage for large N_DATA values.


        if args['dataset'] == 'cifar10': C = 3

        ssim_loss_list = []
        ms_ssim_loss_list = []
        with torch.no_grad():
            for i, data in enumerate(evalloader):
                if (i+1)%24==0 or (i+1)==N_DATA:
                    update_str = '  %s/%s'%(str(i+1),str(N_DATA))
                    if realtime_print:
                        print('  processing %-64s'%(str(update_str)),end='\r')
                x, labels = data
                n_channel = x.shape[1]
                x = x.to(torch.float).to(device=device) 
                
                y = net(x)

                if do_compute_SSIM:
                    with torch.no_grad():
                        target_shape = (n_channel,256,256)
                        oimg = utils.numpy_batch_CHW_resize(x, target_shape, is_pytorch_tensor=True)
                        rimg = utils.numpy_batch_CHW_resize(y, target_shape, is_pytorch_tensor=True)
                        ORIGINAL_IMG_ARRAY.append(oimg)
                        RECONS_IMG_ARRAY.append(rimg)
                        
                        c += 1
                        if c>=120:
                            RECONS_IMG_ARRAY = torch.cat(RECONS_IMG_ARRAY, 0)
                            ORIGINAL_IMG_ARRAY = torch.cat(ORIGINAL_IMG_ARRAY, 0)
                            ssim_loss, ms_ssim_loss = compute_ssim(RECONS_IMG_ARRAY, ORIGINAL_IMG_ARRAY, channel=n_channel,max_val=1.)

                            ssim_loss_list.append(ssim_loss.item())
                            ms_ssim_loss_list.append(ms_ssim_loss.item())
                            ORIGINAL_IMG_ARRAY, RECONS_IMG_ARRAY = [], [] 
                            c = 0

                # print(i, x.shape, y.shape)
                img = y[0].clone().detach().cpu().numpy()
                img = img.transpose(1,2,0)
                img = (img* 255.).astype(np.uint8)
                filename = os.path.join(RECON_DIR, '%s.jpg'%(str(i)))
                # print(img.shape, np.min(img), np.max(img))

                if n_channel==1:
                    img = np.concatenate((img,img,img),axis=2)
                    img = PIL.Image.fromarray(img)
                elif n_channel==3:
                    img = PIL.Image.fromarray(img)
                
                img.save(filename)

                if i+1>=N_DATA: break
        print('\n  Reconstruction done.')

        if do_compute_SSIM:
            RESULT_DICT[int(net.total_iter)] = {'ssim_loss': np.mean(ssim_loss_list),'ms_ssim_loss':np.mean(ms_ssim_loss_list)}
            # print_result_dict(RESULT_DICT)
            TAB_LEVEL = 1
            self.pickle_data(RESULT_DICT, RESULT_DIR, tv=(TAB_LEVEL,0,100), text=None)

        return net # to ensure that we are comparing the results produced by the right model
        
from .evaluation_modules.pytorch_msssim import SSIM, MS_SSIM
def compute_ssim(RECONS_IMG_ARRAY, ORIGINAL_IMG_ARRAY, channel=3,max_val=1.):
    # max_val is 1. if normalized to [0,1]. Otherwise, use 255.
    ssim_module = SSIM(data_range=max_val, size_average=True, channel=channel)
    ms_ssim_module = MS_SSIM(data_range=max_val, size_average=True, channel=channel)
    
    ssim_loss = 1 - ssim_module(RECONS_IMG_ARRAY,ORIGINAL_IMG_ARRAY)
    ms_ssim_loss = 1 - ms_ssim_module(RECONS_IMG_ARRAY,ORIGINAL_IMG_ARRAY)
    return ssim_loss, ms_ssim_loss
 

######################################
# Some result displays
###################################### 

def print_result_dict(RESULT_DICT, this_iter=None):
    print('print_result_dict()')
    if this_iter is None:
        for this_iter, values in RESULT_DICT.items():
            print('iter:%s'%(str(this_iter)))
            for metric, val in values.items():
                print('  %s:%s'%(str(metric),str(val)))
    else:
        print('iter:%s'%(str(this_iter)))
        for metric, val in RESULT_DICT[this_iter].items():
            print('  %s:%s'%(str(metric),str(val)))
    print()

def save_sixteen_images(img_samples, img_dir, title=None):
    plt.figure(figsize=(4,4))
    for i in range(16):
        plt.gcf().add_subplot(4,4,i+1)
        plt.gca().imshow(img_samples[i])
        set_figure_setting(i, 16, set_title=None)
        
        if not title is None:
            if i==0:
                plt.gca().set_title(title)

    plt.savefig(img_dir)
    plt.close()
    return

def plot_results(args):
    from collections import defaultdict

    print('plot_results()')
    _,_, MODEL_DIR , _, _, RESULT_DIR = manage_directories2(args)

    rcon = Reconstructor()
    net = rcon.init_or_load_model(MODEL_DIR, args)
    print('latest net.iter:', net.total_iter)

    TAB_LEVEL = 0 # just for printing
    RESULT_DICT = rcon.load_pickled_data(RESULT_DIR, tv=(TAB_LEVEL,0,100), text=None)
    print_result_dict(RESULT_DICT, this_iter=None)

    iters=[]
    metricd = defaultdict(list)
    for this_iter, values in RESULT_DICT.items():
        iters.append(this_iter)
        for metric, val in values.items():
            metricd[metric].append(val)


    # print(metricd)
    plt.figure()
    plt.gca().plot(iters, metricd['ssim_loss'],c='b',linewidth=0.5,label='ssim_loss')
    plt.gca().plot(iters, metricd['ms_ssim_loss'],c='b',linestyle='--',linewidth=0.5,label='ms_ssim_loss')
    plt.gca().set_xlabel('iter')
    plt.legend()
    plt.gca().twinx()
    plt.gca().plot(iters, metricd['fid_score'],'g',label='fid_score')
    plt.gca().tick_params(axis='y', labelcolor='g')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.85))
    plt.show()
    return