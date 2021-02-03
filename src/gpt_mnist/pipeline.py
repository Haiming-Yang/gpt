import time, os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from .utils import folder_check, Logger, save_ckpt_dir
import src.gpt_mnist.data as ud

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def entry(args):
    if args['mode'] == 'training':
        training(args)
    elif args['mode'] == 'view_losses':
        view_losses(args)
    elif args['mode'] == 'generate_samples':
        generate_samples(args)
    elif args['mode'] == 'apply_mnist':
        apply_to_mnist(args)
    elif args['mode'] == 'finetune':
    	from src.gpt_mnist.finetuner.finetune import redirect
    	redirect(args)
    elif args['mode'] == 'heatmaps':
        heatmaps(args)
    elif args['mode'] == 'heatmapsGC':
        heatmaps_of_gen_config(args)
    elif args['mode'] == 'gen_dist':
        gen_distribution(args)
    else:
        raise RuntimeError('Invalid mode specified.')        

########################################################################
# MAIN TRAINING
########################################################################

def manually_parse_boolean(x):
    # just to be safe
    if x == '0' or x =='False':
        x = False
    elif x == '1' or x =='True':
        x = True
    else:
        raise RuntimeError('Be careful with parsing boolean. Play safe.')
    return x

def apply_to_mnist(args):
    PROJECT_ID = args['PROJECT_ID']

    CKPT_DIR, PROJECT_DIR, MODEL_DIR, LOGGER_DIR, load_model = folder_check(PROJECT_ID, CKPT_DIR='checkpoint')
    
    import torchvision
    print('apply to mnist')
    dataset = torchvision.datasets.MNIST('data', train=True, download = False)
    x_batch = []
    for i in range(10):
        img,y = dataset.__getitem__(i)
        x = np.array(img)
        x_batch.append([x])
    x_batch = torch.tensor(x_batch).to(device=device).to(torch.float)

    from .sampler import Pytorch_GPT_MNIST_Sampler
    samp = Pytorch_GPT_MNIST_Sampler(compenv_mode=None, growth_mode=None)

    from .model import ResGPTNet34
    net = ResGPTNet34(nG0=samp.gen.nG0, Nj=samp.gen.N_neighbor)
    net.to(device=device)
    net = torch.load(MODEL_DIR)
    net.eval()

    y, yg, ys = net(x_batch)

    evaluation_visual_results(x_batch, y, yg, ys, None, None, None, GPTGenerator=samp.gen, save_dir=None, transpose=True, )


def gen_distribution(args):
    print('plot generator distributions...')
    from .sampler import Pytorch_GPT_MNIST_Sampler
    samp = Pytorch_GPT_MNIST_Sampler(compenv_mode=None, growth_mode=None)

    N_SAMPLES_PER_CLASS = 240
    N_CLASS = 10

    plt.figure(figsize=(7,5))
    plt.gcf().add_subplot(111)
    dist, dist_s = np.zeros((28,28)), np.zeros((28,28))
    for c in range(N_CLASS):
        for j in range(N_SAMPLES_PER_CLASS):
            _, _, yg0, ys0 = samp.get_sample_batch([c], device=None)
            yg0 = yg0[0].clone().detach().cpu().numpy()>0
            dist = dist+yg0/(N_SAMPLES_PER_CLASS*N_CLASS)
        
    dg = plt.gca().imshow(dist, label='%s'%(str(c)), cmap='hot',vmin=0., )
    plt.gcf().colorbar(dg,ax=plt.gca())
    plt.gca().set_title('yg dist')
    plt.show()        

def training(args):  
    print('\ntraining()\n')
    for x,y in args.items():
        print('%-14s:%s %s'%(str(x),str(y), str(type(y))))

    PROJECT_ID = args['PROJECT_ID']
    batch_size = args['batch_size']
    N_epoch = args['N_EPOCH']
    n_iter_per_epoch = args['N_PER_EPOCH']
    track_every_epoch = manually_parse_boolean(str(args['track_every_epoch']))
    regularizations = args['regularizations']
    n_eval_epoch = args['n_eval_epoch']
    n_eval_batch = args['n_eval_batch']
    realtime_print = manually_parse_boolean(str(args['realtime_print']))
    debug_target = manually_parse_boolean(str(args['debug_target']))
    print('arguments adjusted!\n')
    
    # Set target losses by observing how losses change when more predictions are set to correct.
    if not debug_target: 
        TARGET_LOSSES = [1e-5,0.0002,0.0002]
        SOFT_TARGET_LOSSES = [1e-5,0.001,0.001]
    else: 
        TARGET_LOSSES = [0.1,1.4,1.4] # FOR DEBUGGING
        SOFT_TARGET_LOSSES = [0.4,1.6,1.5]        

    print(TARGET_LOSSES)
    print(SOFT_TARGET_LOSSES)

    CKPT_DIR, PROJECT_DIR, MODEL_DIR, LOGGER_DIR, load_model = folder_check(PROJECT_ID, CKPT_DIR='checkpoint')

    logger = Logger()

    from .sampler import Pytorch_GPT_MNIST_Sampler
    samp = Pytorch_GPT_MNIST_Sampler(compenv_mode=None, growth_mode=None)

    from .model import ResGPTNet34
    net = ResGPTNet34(nG0=samp.gen.nG0, Nj=samp.gen.N_neighbor)
    net.to(device=device)

    if load_model: 
        net = torch.load(MODEL_DIR)
        print('loading model at %s iter.'%(str(net.tracker['iter'])))
        logger = logger.load_pickled_data(LOGGER_DIR, tv=(0,0,None), text=None)

    ###################################################################
    # different weight for different losses
    criterion = nn.CrossEntropyLoss()

    gc_weights = np.ones(shape=(samp.gen.nG0+1))
    gc_weights[0]= 1e-2 # 1e-4
    gc_weights = torch.tensor(gc_weights).to(device=device).to(torch.float)
    criterion_gc = nn.CrossEntropyLoss(weight=gc_weights)

    gt_weights = np.ones(shape=(samp.gen.N_neighbor))
    gt_weights[0]= 1e-2
    gt_weights = torch.tensor(gt_weights).to(device=device).to(torch.float)
    criterion_gt = nn.CrossEntropyLoss(weight=gt_weights)
    ###################################################################

    soft_lr = 0.0001
    if not net.setting['softened_learning_rate']:
        learning_rate = 0.001
    else:
        learning_rate = soft_lr
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.5,0.999),weight_decay=1e-5)

    if regularizations is None:
        lambda1 = lambda2 = 1
    elif regularizations == 'output_size': # does not seem to work
        lambda1 = lambda2 = (28**2)/10 # 10 for 10 classes

    start = time.time()
    n_eval, n_correct = None, None
    for ne in range(N_epoch):
        for i in range(n_iter_per_epoch):
            net.train()
            net.zero_grad()
            
            x, y0, yg0, ys0 = samp.get_sample_batch_uniform_random(batch_size=batch_size, device=device)
            y, yg, ys = net(x)
            # print(y.shape, yg.shape, ys.shape)

            class_pred_loss = criterion(y,y0)
            gen_config_loss = criterion_gc(yg, yg0)
            gen_transform_loss = criterion_gt(ys, ys0) 
            loss = class_pred_loss + lambda1 * gen_config_loss + lambda2 * gen_transform_loss

            loss.backward()
            optimizer.step()

            net.tracker['iter'] += 1
            logger.iter_array.append(net.tracker['iter'])
            logger.loss_array['class_pred_loss'].append(class_pred_loss.item())
            logger.loss_array['gen_config_loss'].append(gen_config_loss.item())
            logger.loss_array['gen_transform_loss'].append(gen_transform_loss.item())


            if not net.setting['softened_learning_rate']:
                SOFT_TARGET1 = class_pred_loss.item()<=SOFT_TARGET_LOSSES[0] 
                SOFT_TARGET2 = gen_config_loss.item()<=SOFT_TARGET_LOSSES[1]
                SOFT_TARGET3 = gen_transform_loss.item()<=SOFT_TARGET_LOSSES[2]
                ST = SOFT_TARGET1 and SOFT_TARGET2 and SOFT_TARGET3
                if ST: 
                    net.setting['softened_learning_rate'] = True
                    optimizer.defaults['lr'] = soft_lr
                    _, MODEL_CKPT_DIR = save_ckpt_dir(PROJECT_DIR, model=net)
                    torch.save(net, MODEL_CKPT_DIR[:-4]+'.soft.net')

            # ALL MUST BE ACHIEVED FOR EARLY STOP
            EARLY_STOP1 = class_pred_loss.item()<= TARGET_LOSSES[0]
            EARLY_STOP2 = gen_config_loss.item()<= TARGET_LOSSES[1]
            EARLY_STOP3 = gen_transform_loss.item()<=TARGET_LOSSES[2]
            EARLY_STOP = EARLY_STOP1 and EARLY_STOP2 and EARLY_STOP3

            if realtime_print:
                status = 'NORMAL'
                if EARLY_STOP: status = 'EARLY STOP'
                if net.setting['softened_learning_rate']: status = status+'.SOFT_lr'
                if i%4==0 or i+1==n_iter_per_epoch or EARLY_STOP:
                    epoch_update = '%s'%('epoch: %s/%s iter:%s/%s'%(str(ne+1),str(N_epoch),str(i+1),str(n_iter_per_epoch)))
                    l1,l2,l3 = round(class_pred_loss.item(),4),round(gen_config_loss.item(),4),round(gen_transform_loss.item(),4)
                    print('%-96s'%('%s losses:%s,%s,%s [%s] lr:%s'%(epoch_update,l1,l2,l3, status, str(optimizer.defaults['lr']))),end='\r')
            if EARLY_STOP: break

        
        if track_every_epoch:
            IMG_DIR, MODEL_CKPT_DIR = save_ckpt_dir(PROJECT_DIR, model=net)
            torch.save(net, MODEL_CKPT_DIR)

            with torch.no_grad():
                net.eval()
                n_eval, n_correct = 0,0
                for i_eval in range(n_eval_epoch):
                    x, y0, yg0, ys0 = samp.get_sample_batch_uniform_random(batch_size=n_eval_batch, device=device)
                    y, yg, ys = net(x)
                    
                    for j_eval, y_pred in enumerate(y):
                        pred_class = int(torch.argmax(y_pred))
                        true_class = int(y0[j_eval])
                        if true_class==pred_class: n_correct+=1
                        n_eval+=1

                    if i_eval==0:
                        evaluation_visual_results(x, y,yg,ys, y0, yg0, ys0, GPTGenerator=samp.gen, save_dir=IMG_DIR)
            logger.records['by_iter'][net.tracker['iter']] = {
                'n_correct': n_correct,
                'n_eval': n_eval,
            }
        if EARLY_STOP: break

    end = time.time()

    if EARLY_STOP: print('\n\ngood! early stop activated!')

    elapsed = end - start
    if (not n_correct is None) and (not n_eval is None):
        acc = round(n_correct/n_eval,3)
    else:
        acc = None

    print('\ntime taken %s[s] = %s [min] = %s [hr], latest acc:%s/%s=%s'%(
        str(round(elapsed,1)), str(round(elapsed/60.,1)), str(round(elapsed/3600.,1)),
        str(n_correct),str(n_eval),str(acc))
    )
    logger.n_th_run += 1
    logger.records[logger.n_th_run] = {
        'time': elapsed,
        'N_epoch': N_epoch,
        'n_iter_per_epoch': n_iter_per_epoch,
        'batch_size':batch_size,
    }

    torch.save(net, MODEL_DIR)
    torch.save(net, MODEL_DIR[:-4]+'.backup.net')
    logger.pickle_data(logger, LOGGER_DIR, tv=(0,0,None), text=None)


def evaluation_visual_results(x, y,yg,ys, y0, yg0, ys0, GPTGenerator, save_dir, transpose=False, display_gen=False):
    batch_size = x.shape[0]

    if not transpose:
        plt.figure(figsize=(8,2*batch_size))
    else:
        plt.figure(figsize=(int(batch_size-1),5))

    for i in range(batch_size):
        board_shape = (GPTGenerator.L,GPTGenerator.L)

        # 1. the image
        img = x[i].cpu().clone().detach().numpy()[0]

        # 2. grow the image from ground truth seeds for double-checking
        T_MAX = 8 
        if not y0 is None:
            gens0 = yg0[i].clone().detach().cpu().numpy() + 1
            s0 = ys0[i].clone().detach().cpu().numpy() + 1
            board = np.ones(shape=board_shape)
            seeds = ud.code_generator(gens0, s0, GPTGenerator.nBSG)
            board = seeds
            img1 = GPTGenerator.develop(T_MAX, board, 
                compenv_mode=GPTGenerator.compenv_mode, growth_mode=GPTGenerator.growth_mode)
            img1 = 1.*(img1>GPTGenerator.nBSG)

        # 3. predicted outputs
        gens = yg[i].clone().detach().cpu().numpy()
        gens = np.argmax(gens,axis=0) + 1
        s = ys[i].clone().detach().cpu().numpy()
        s = np.argmax(s,axis=0) + 1
        seeds_pred = ud.code_generator(gens, s, GPTGenerator.nBSG)
        seeds_pred = np.array(seeds_pred)
        board_pred = seeds_pred
        img_pred = GPTGenerator.develop(T_MAX, board_pred, 
            compenv_mode=GPTGenerator.compenv_mode, growth_mode=GPTGenerator.growth_mode)
        img_pred = 1.*(img_pred>GPTGenerator.nBSG)

        if not transpose:
            plt.gcf().add_subplot(batch_size,3,1 + i*3)
            plt.gca().imshow(img, cmap='gray')
            if i==0: plt.gca().set_title('img')
            plot_setting(i, batch_size)

            if not y0 is None:
                plt.gcf().add_subplot(batch_size,3,2 + i*3)
                plt.gca().imshow(img1, cmap='gray')
                if i==0: plt.gca().set_title('grown y0')
                plot_setting(i, batch_size)

            plt.gcf().add_subplot(batch_size,3,3 + i*3)
            plt.gca().imshow(img_pred, cmap='gray')
            plot_setting(i, batch_size)
            if i==0: plt.gca().set_title('grown y_predv')
        else:
            plt.gcf().add_subplot(4,batch_size,1+i)
            plt.gca().imshow(img, cmap='gray')

            if i==0: plt.gca().set_title('img')
            plot_setting(i, batch_size)

            if not y0 is None:
                plt.gcf().add_subplot(4,batch_size,1+i+batch_size)
                plt.gca().imshow(img1, cmap='gray')
                if i==0: plt.gca().set_title('grown y0')
                plot_setting(i, batch_size)

            plt.gcf().add_subplot(4,batch_size,1+i+2*batch_size)
            plt.gca().imshow(img_pred, cmap='gray')
            plot_setting(i, batch_size)
            if i==0: plt.gca().set_title('grown y_predv')

            plt.gcf().add_subplot(4,batch_size,1+i+3*batch_size)
            plt.gca().imshow(img, cmap='gray')
            plt.gca().imshow(gens>1, cmap='bwr',vmin=-1,vmax=1,alpha=0.8)
            plot_setting(i, batch_size)


    if save_dir is None:
        plt.show()
    else:
        plt.savefig(save_dir)

def plot_setting(i, batch_size):
    if i+1==batch_size:
        plt.gca().set_yticks([])
    elif i==0:
        plt.gca().set_xticks([])  
    else:
        plt.gca().set_xticks([]);plt.gca().set_yticks([])

def view_losses(args):
    PROJECT_ID = args['PROJECT_ID']
    average_n_loss = args['average_n_loss']
    iter0_n_loss = args['iter0_n_loss']

    _,_, _, LOGGER_DIR, _ = folder_check(PROJECT_ID, CKPT_DIR='checkpoint')

    logger = Logger()
    logger = logger.load_pickled_data(LOGGER_DIR, tv=(0,0,None), text=None)
    for kth_run, record_dict in logger.records.items():
        if kth_run=='by_iter': continue
        print('kth_run:\n',record_dict)
        elapsed = record_dict['time']
        print(' time taken %s[s] = %s [min] = %s [hr]'%(
            str(round(elapsed,1)), str(round(elapsed/60.,1)), str(round(elapsed/3600.,1))))

    for eval_iter,r in logger.records['by_iter'].items():
        n_correct = r['n_correct']
        n_eval = r['n_eval']
        print('iter:%s acc:%s/%s=%s'%(str(eval_iter),str(n_correct),
            str(n_eval),str(round(n_correct/n_eval,3))))

    if iter0_n_loss is not None:
        logger.iter_array = logger.iter_array[iter0_n_loss:]
        for x in logger.loss_array:
            logger.loss_array[x] = logger.loss_array[x][iter0_n_loss:]

    if average_n_loss is None:
        plt.plot(logger.iter_array, logger.loss_array['class_pred_loss'],label='class_pred_loss')
        plt.plot(logger.iter_array, logger.loss_array['gen_config_loss'], label='gen_config_loss')
        plt.plot(logger.iter_array, logger.loss_array['gen_transform_loss'],label='gen_transform_loss')
    else:
        iter_arrays = np.array(logger.iter_array).reshape(-1,average_n_loss)[:,-1]
        l1 = np.array(logger.loss_array['class_pred_loss']).reshape(-1,average_n_loss)
        l1 = np.mean(l1,axis=1)
        plt.plot(iter_arrays, l1, label='class_pred_loss')
        l2 = np.array(logger.loss_array['gen_config_loss']).reshape(-1,average_n_loss)
        l2 = np.mean(l2,axis=1)
        plt.plot(iter_arrays, l2, label='gen_config_loss')
        l3 = np.array(logger.loss_array['gen_transform_loss']).reshape(-1,average_n_loss)
        l3 = np.mean(l3,axis=1)
        plt.plot(iter_arrays, l3,label='gen_transform_loss')


    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

def generate_samples(args):
    PROJECT_ID = args['PROJECT_ID']
    n_eval_batch = args['n_eval_batch']
    random_batch = manually_parse_boolean(str(args['random_batch']))

    CKPT_DIR, PROJECT_DIR, MODEL_DIR, LOGGER_DIR, load_model = folder_check(PROJECT_ID, CKPT_DIR='checkpoint')
    logger = Logger()

    from .sampler import Pytorch_GPT_MNIST_Sampler
    samp = Pytorch_GPT_MNIST_Sampler(compenv_mode=None, growth_mode=None)

    from .model import ResGPTNet34
    net = ResGPTNet34(nG0=samp.gen.nG0, Nj=samp.gen.N_neighbor)
    net.to(device=device)

    if load_model: 
        net = torch.load(MODEL_DIR)
        print('loading model at %s iter.'%(str(net.tracker['iter'])))
        logger = logger.load_pickled_data(LOGGER_DIR, tv=(0,0,None), text=None)
    else:
    	print('model not found? Check %s. '%(str(MODEL_DIR)))
    
    net.output_mode=None
    if random_batch:
        x, y0, yg0, ys0 = samp.get_sample_batch_uniform_random(batch_size=n_eval_batch, device=device)
    else:
        x, y0, yg0, ys0 = samp.get_sample_batch(range(10), device=device)
    y, yg, ys = net(x)

    evaluation_visual_results(x, y, yg, ys, y0, yg0, ys0, GPTGenerator=samp.gen, save_dir=None, transpose=True, )


def generate_generator_dist(args):
    print('generate_generator_dist')

########################################################################
# HEATMAPS
########################################################################

def heatmaps(args):
    print('heatmaps')
    PROJECT_ID = args['PROJECT_ID']

    CKPT_DIR, PROJECT_DIR, MODEL_DIR, LOGGER_DIR, load_model = folder_check(PROJECT_ID, CKPT_DIR='checkpoint')
    XAI_DIR = os.path.join(PROJECT_DIR,'XAI')
    if not os.path.exists(XAI_DIR): os.mkdir(XAI_DIR)

    from .sampler import Pytorch_GPT_MNIST_Sampler
    samp = Pytorch_GPT_MNIST_Sampler(compenv_mode=None, growth_mode=None)

    from .model import ResGPTNet34
    net = ResGPTNet34(nG0=samp.gen.nG0, Nj=samp.gen.N_neighbor)
    net = torch.load(MODEL_DIR)
    net.output_mode = 'prediction_only'
    net.to(device=device)
    net.eval()

    x, y0, yg0, ys0 = samp.get_sample_batch(class_indices=np.array(range(10)), device=device)
    x.requires_grad=True

    attrs = {}
    SAVE_DIR = os.path.join(XAI_DIR, 'heatmaps.y0.jpeg')

    from captum.attr import LayerGradCam, Deconvolution, GuidedBackprop # ShapleyValueSampling

    xai = LayerGradCam(net, net.channel_adj)
    attr = xai.attribute(x, target=y0).clone().detach().cpu().numpy()
    attrs['gradCAM'] = attr

    xai = Deconvolution(net)
    attr = xai.attribute(x, target=y0).clone().detach().cpu().numpy()
    attrs['deconv'] = attr

    xai = GuidedBackprop(net)
    attr = xai.attribute(x, target=y0).clone().detach().cpu().numpy()
    attrs['GuidedBP'] = attr

    arrange_heatmaps(x.clone().detach().cpu().numpy() , attrs, save_dir=SAVE_DIR)

def arrange_heatmaps(x, attrs, save_dir=None):
    batch_size = x.shape[0]
    n_column = 1 + len(attrs)

    plt.figure(figsize=(2*n_column,2*batch_size))
    for i in range(batch_size):
        plt.gcf().add_subplot(batch_size,n_column,i*n_column+1)
        plt.gca().imshow(x[i][0], cmap='gray')
        plot_setting(i, batch_size)
        
        for j, (attr_name, attr) in enumerate(attrs.items()):
            h = attr[i][0]
            h_abs = np.abs(h)
            h = h/np.max(h_abs)
            plt.gcf().add_subplot(batch_size,n_column,i*n_column+1 + j+1)
            plt.gca().imshow(h, cmap='bwr', vmin=-1,vmax=1)
            plot_setting(i, batch_size)

            if i==0:
                plt.gca().set_title('%s'%(str(attr_name)))

    if save_dir is None:
        plt.show()
    else:
        print('saving figure to %s'%(str(save_dir)))
        plt.savefig(save_dir)

class WrapperNet(nn.Module):
    """To wrap a network and redirect the forward() function
    so that it becomes compatible with captum"""
    def __init__(self, main_net, output_mode='yg_pixel', **kwargs):
        super(WrapperNet, self).__init__()
        self.main_net = main_net
        self.output_mode = output_mode

        if self.output_mode in ['yg_pixel','ys_pixel']:
            # can set spatial_coords later, but for clarity better identify earlier pixel of interest. 
            self.spatial_coords = kwargs['spatial_coords'] 
        else:
            raise RuntimeError('Invalid WrapperNet output mode.')

    def forward(self, x):
        assert(x.shape[0]==1) # this only accepts 1 sample data, not batches of data
        y, yg, ys = self.main_net(x)
        if self.output_mode == 'yg_pixel':
            idx, idy = self.spatial_coords
            out = yg[0:1,:,idx,idy]
        elif self.output_mode == 'ys_pixel':
            idx, idy = self.spatial_coords
            out = ys[0:1,:,idx,idy]
        return out
        
def heatmaps_of_gen_config(args):
    print('heatmaps_of_gen_config')
    PROJECT_ID = args['PROJECT_ID']

    n_class = 10

    CKPT_DIR, PROJECT_DIR, MODEL_DIR, LOGGER_DIR, load_model = folder_check(PROJECT_ID, CKPT_DIR='checkpoint')
    XAI_DIR = os.path.join(PROJECT_DIR,'XAI')
    if not os.path.exists(XAI_DIR): os.mkdir(XAI_DIR)

    from .sampler import Pytorch_GPT_MNIST_Sampler
    samp = Pytorch_GPT_MNIST_Sampler(compenv_mode=None, growth_mode=None)

    from .model import ResGPTNet34
    net = torch.load(MODEL_DIR)
    net = ResGPTNet34(nG0=samp.gen.nG0, Nj=samp.gen.N_neighbor)
    net = torch.load(MODEL_DIR)
    net.output_mode = None
    net.to(device=device)
    net.eval()

    for i in range(n_class): # 1 class, 1 example to print
        x, y0, yg0, ys0 = samp.get_sample_batch(class_indices=[i], device=device)
        x.requires_grad=True
        y, yg, ys = net(x)

        img = x[0][0].clone().detach().cpu().numpy()

        # 1. the groundtruth generators config
        gens0 = yg0[0].clone().detach().cpu().numpy() + 1
        ss0 = ys0[0].clone().detach().cpu().numpy() + 1
        
        # 2. the predicted generators config
        gens_pred = yg[0].clone().detach().cpu().numpy()
        gens_pred = np.argmax(gens_pred,axis=0) + 1 # (28,28)      
        ss_pred = ys[0].clone().detach().cpu().numpy()
        ss_pred = np.argmax(ss_pred,axis=0) + 1

        # NOTE: if the predictions are accurate, both outputs will look the same
        GENS = [gens0, gens_pred]
        GENS_PY = [yg0, torch.argmax(yg,dim=1)]
        IMGTAGS = ['yg0','yg']
        for gens, gens_py, img_tag in zip(GENS, GENS_PY, IMGTAGS):
            attrs, genss = compute_attr_non_emtpy_gens(x, net, gens, gens_py)
            save_dir = os.path.join(XAI_DIR,str('%s.%s.jpg'%(str(i),str(img_tag))))
            arrange_heatmapsGC(img, attrs, genss, save_dir=save_dir)


        SS = [ss0,ss_pred]
        SS_PY = [ys0, torch.argmax(ys,dim=1)]
        IMGsTAGS = ['ys0','ys']
        for gens, ss, ss_py, img_tag in zip(GENS, SS, SS_PY, IMGsTAGS):
            attrTRs, TRs = compute_attr_transformations_of_non_emtpy_gens(x, net, ss, ss_py, gens)
            save_dir = os.path.join(XAI_DIR,str('%s.%s.jpg'%(str(i),str(img_tag))))
            arrange_heatmapsGC(img, attrTRs, TRs, save_dir=save_dir)

def compute_attr_non_emtpy_gens(x, net, gens, gens_py):
    """
    Compute different heatmaps from non-empty-generator yg pixels of a single image sample, assuming the network output y,yg,ys = net(x) (see ResGPTNet34)
    
    args:
        x is pytorch tensor of size (batch_size, channel, height, width) = (1,1,28,28).
        net. The neural network, with architecture like ResGPTNet34.
        gens are array of generators. A generator is either 1,2,...,nG0
        gens_py is gens, but in pytorch tensor representation (include batch_size)

    return
       attrs: heatmaps computed from pixelwise generator config. 
       genns: the values of the relevant yg pixel
    """

    idxs, idys = np.where(gens>1) # only interested in pixels predicted to have non-empty generators
    n_gens_pred = len(idxs)

    attrs, genss = {},{}
    for idx, idy in zip(idxs,idys):
        yg_target_pix = gens_py[0,idx,idy]
        
        attrs[(idx,idy)] = {}
        genss[(idx,idy)] = gens[idx,idy]

        attrs[(idx,idy)]['gradCAM'] = compute_attr_one_pixel_target(x, net, (idx,idy), 'gradCAM', target=yg_target_pix)
        attrs[(idx,idy)]['deconv'] = compute_attr_one_pixel_target(x, net, (idx,idy), 'deconv', target=yg_target_pix)
        attrs[(idx,idy)]['GuidedBP'] = compute_attr_one_pixel_target(x, net, (idx,idy), 'GuidedBP', target=yg_target_pix)
    return attrs, genss

def compute_attr_transformations_of_non_emtpy_gens(x, net, ss, ss_py, gens): 
    """
    Similar to compute_attr_non_emtpy_gens(), but for ys pixels.
    """
    idxs, idys = np.where(gens>1) #only interested in pixels with non-empty generators
    n_gens_pred = len(idxs)

    attrTRs, TRs = {},{}
    for idx, idy in zip(idxs,idys):
        ys_target_pix = ss_py[0,idx,idy]
        
        attrTRs[(idx,idy)] = {}
        TRs[(idx,idy)] = ss[idx,idy]

        attrTRs[(idx,idy)]['gradCAM'] = compute_attr_one_pixel_target(x, net, (idx,idy), 'gradCAM', target=ys_target_pix, wrapper_output='ys_pixel')
        attrTRs[(idx,idy)]['deconv'] = compute_attr_one_pixel_target(x, net, (idx,idy), 'deconv', target=ys_target_pix, wrapper_output='ys_pixel')
        attrTRs[(idx,idy)]['GuidedBP'] = compute_attr_one_pixel_target(x, net, (idx,idy), 'GuidedBP', target=ys_target_pix, wrapper_output='ys_pixel')
    return attrTRs, TRs

def compute_attr_one_pixel_target(x, net, spatial_coords, method, **kwargs):
    # x is a single input tensor, i.e. shape (batch_size,channel_size,H,W)=(1,1,28,28)
    from captum.attr import LayerGradCam, Deconvolution, GuidedBackprop

    if 'target' in kwargs:
        target = kwargs['target']
    
    if 'wrapper_output' in kwargs:
        output_mode = kwargs['wrapper_output']
    else:
        output_mode = 'yg_pixel'

    idx,idy = spatial_coords
    wnet = WrapperNet(net, output_mode=output_mode, spatial_coords=(idx,idy))
    
    if method=='gradCAM':
        xai = LayerGradCam(wnet, wnet.main_net.channel_adj)
    elif method=='deconv':
        xai = Deconvolution(wnet)
    elif method=='GuidedBP':
        xai = GuidedBackprop(wnet)
    
    if method in ['gradCAM', 'deconv', 'GuidedBP']:
        attr = xai.attribute(x, target=target )
    elif method == 'layerAct':
        attr = xai.attribute(x)

    attr = attr[0][0].clone().detach().cpu().numpy()
    return attr


def arrange_heatmapsGC(img, attrs, genss, save_dir=None):
    n_gens = n_row = len(attrs) 

    if n_gens>0:
        for i, ((idx,idy), attr_dict) in enumerate(attrs.items()):
            n_column = len(attr_dict)+1
            if i==0:
                plt.figure(figsize=(2*n_column,2*n_row))
            
            genpos = np.zeros(shape=img.shape)
            genpos[(idx,idy)] = 1.

            plt.gcf().add_subplot(n_row,n_column,1+i*n_column)
            plt.gca().imshow(img, cmap='gray')
            plt.gca().imshow(genpos, cmap='bwr',vmin=-1.,vmax=1.,alpha=0.8)
            plt.gca().set_title('%s'%(str(genss[(idx,idy)])))
            plot_setting(i, n_row)

            # plt.imshow(,alpha=0.5)
            for j, (attr_name, attr) in enumerate(attr_dict.items()):    
                h_abs = np.max(np.abs(attr))
                h = attr/h_abs

                plt.gcf().add_subplot(n_row,n_column,1+i*n_column +(j+1))
                plt.imshow(h, cmap='bwr',vmin=-1.,vmax=1.)
                plt.gca().imshow(img, cmap='gray', alpha=0.2)

                plot_setting2(i, j+1, n_row, n_column)
                if i==0:
                    plt.gca().set_title(attr_name)
    else:
        plt.figure()
        plt.imshow(img, cmap='gray')

    if save_dir is None:
        plt.show()
    else:
        print('save figure to %s'%(str(save_dir)))
        plt.savefig(save_dir)
    plt.close()

def plot_setting2(i,j, n_row,n_column):
    if i+1==n_row and j==0:
        plt.gca().set_yticks([])
    elif i==0 and j==0:
        plt.gca().set_xticks([])  
    else:
        plt.gca().set_xticks([]);plt.gca().set_yticks([])

