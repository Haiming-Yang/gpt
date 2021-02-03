import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from ..utils import folder_check, Logger,  save_ckpt_dir
import src.gpt_mnist.pipeline as pp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def do_finetune(args):
    print('Finetuning from finetune_Simple0001.py')
    SOURCE_PROJECT_ID = 'Simple0001' #from treasure trove

    for x,y in args.items():
        print('%-14s:%s %s'%(str(x),str(y), str(type(y))))
    FINETUNE_ID = args['FINETUNE_ID']
    batch_size = args['batch_size']
    N_epoch = args['N_EPOCH']
    n_iter_per_epoch = args['N_PER_EPOCH']
    load_from_trove = pp.manually_parse_boolean(str(args['load_from_trove']))
    realtime_print = pp.manually_parse_boolean(str(args['realtime_print']))
    TARGET_LOSS1 = 1e-4
    TARGET_LOSS2 = 0.02
    TARGET_LOSS3 = 0.02
    learning_rate = 1e-4
    print('arguments adjusted!\n')

    n_eval_epoch = 10
    n_eval_batch = 10
    track_every_epoch = 1
    average_n_iters = 8
    TARGET_AVG_LOSSES = [TARGET_LOSS1,TARGET_LOSS2,TARGET_LOSS3]
    
    print('Target directories:')
    CKPT_DIR, PROJECT_DIR, MODEL_DIR, LOGGER_DIR, _ = folder_check(FINETUNE_ID, CKPT_DIR='checkpoint')
    
    if load_from_trove:
        print('Scanning the _treasure_trove to load from:')
        TROVE_DIR, TROVE_PROJECT_DIR, TROVE_MODEL_DIR, TROVE_LOGGER_DIR, _ = folder_check(SOURCE_PROJECT_ID, CKPT_DIR='_treasure_trove')    
        samp, net, logger = setup_main_components(TROVE_MODEL_DIR, TROVE_LOGGER_DIR)
    else:
        samp, net, logger = setup_main_components(MODEL_DIR, LOGGER_DIR)

    lambda1,lambda2 = 1.,1.
    lmon = LossMonitor(average_n_iters=average_n_iters)
    
    ###################################################################
    criterion = nn.CrossEntropyLoss()
    criterion_gc = nn.CrossEntropyLoss()
    criterion_gt = nn.CrossEntropyLoss()
    ###################################################################

    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.5,0.999),weight_decay=1e-5)

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
            l1,l2,l3 = class_pred_loss.item(),gen_config_loss.item(),gen_transform_loss.item()
            logger.loss_array['class_pred_loss'].append(l1)
            logger.loss_array['gen_config_loss'].append(l2)
            logger.loss_array['gen_transform_loss'].append(l3)
            l1_avg, l2_avg, l3_avg = lmon.compute_current_running_avg(l1,l2,l3)

            AVG_TARGET_ACHIEVED = [l1_avg<TARGET_AVG_LOSSES[0],l2_avg<TARGET_AVG_LOSSES[1],l3_avg<TARGET_AVG_LOSSES[2]]            
            AVG_TARGET_ACHIEVED = np.all(AVG_TARGET_ACHIEVED)
            
            if realtime_print:
                status = 'NORMAL'
                if AVG_TARGET_ACHIEVED: status = 'AVG_TARGET_ACHIEVED'
                if i%4==0 or i+1==n_iter_per_epoch:
                    epoch_update = '%s'%('epoch: %s/%s iter:%s/%s'%(str(ne+1),str(N_epoch),str(i+1),str(n_iter_per_epoch)))
                    l1_avg, l2_avg, l3_avg = round(l1_avg,4),round(l2_avg,4),round(l3_avg,4)
                    print('%-96s'%('%s avg losses:%s,%s,%s [%s] lr:%s'%(epoch_update,l1_avg, l2_avg, l3_avg, status, str(optimizer.defaults['lr']))),end='\r')

            if AVG_TARGET_ACHIEVED: break
        if AVG_TARGET_ACHIEVED: break

    end = time.time()
    elapsed = end - start

    if AVG_TARGET_ACHIEVED: print('\n<<AVG_TARGET_ACHIEVED>>\n')

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
                pp.evaluation_visual_results(x, y,yg,ys, y0, yg0, ys0, GPTGenerator=samp.gen, save_dir=IMG_DIR)
    logger.records['by_iter'][net.tracker['iter']] = {
        'n_correct': n_correct,
        'n_eval': n_eval,
    }
    acc = round(n_correct/n_eval,3)

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

def setup_main_components(MODEL_DIR, LOGGER_DIR):
    from ..sampler import Pytorch_GPT_MNIST_Sampler
    samp = Pytorch_GPT_MNIST_Sampler(compenv_mode=None, growth_mode=None)

    from ..model import ResGPTNet34
    net = ResGPTNet34(nG0=samp.gen.nG0, Nj=samp.gen.N_neighbor)
    net.to(device=device)

    net = torch.load(MODEL_DIR)
    print('loading model at %s iter.'%(str(net.tracker['iter'])))

    logger = Logger()
    logger = logger.load_pickled_data(LOGGER_DIR, tv=(0,0,None), text=None)
    return samp, net, logger

class LossMonitor(object):
    def __init__(self, average_n_iters=8):
        super(LossMonitor, self).__init__()
        self.l1_running_list = []
        self.l2_running_list = []
        self.l3_running_list = []

        self.average_n_iters = average_n_iters
        self.n_iter = 0

    def compute_current_running_avg(self,l1,l2,l3):
        # l1,l2,l3 are losses, scalar
        if self.n_iter<self.average_n_iters:
            self.n_iter += 1
        else:
            self.l1_running_list.pop(0)
            self.l2_running_list.pop(0)
            self.l3_running_list.pop(0)            
        self.l1_running_list.append(l1)
        self.l2_running_list.append(l2)
        self.l3_running_list.append(l3)

        l1_avg = np.mean(self.l1_running_list)
        l2_avg = np.mean(self.l2_running_list)
        l3_avg = np.mean(self.l3_running_list)
        return l1_avg, l2_avg, l3_avg