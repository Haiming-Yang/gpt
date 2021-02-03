import time, os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from ..utils import folder_check, Logger,  save_ckpt_dir
import src.gpt_mnist.pipeline as pp

from .finetune_NSCC_NICE1 import setup_main_components, LossMonitor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def do_finetune(args):
    print('Finetuning from finetune_NSCC_NICE2.py')
    SOURCE_PROJECT_ID = 'NSCC_NICE2' #from treasure trove

    for x,y in args.items():
        print('%-14s:%s %s'%(str(x),str(y), str(type(y))))
    FINETUNE_ID = args['FINETUNE_ID']
    batch_size = args['batch_size']
    N_epoch = args['N_EPOCH']
    n_iter_per_epoch = args['N_PER_EPOCH']
    load_from_trove = pp.manually_parse_boolean(str(args['load_from_trove']))
    realtime_print = pp.manually_parse_boolean(str(args['realtime_print']))
    learning_rate = args['learning_rate'] if (args['learning_rate'] is not None) else 1e-5
    print('arguments adjusted!\n')

    TARGET_LOSS1 = 2e-4
    TARGET_LOSS2 = 2e-4
    TARGET_LOSS3 = 2e-4
    n_eval_epoch = 10
    n_eval_batch = 10
    track_every_epoch = 1
    average_n_iters = 8
    class_probability_distribution = [1,1,1,1,1 ,1,1,1,6,6]


    class_probability_distribution = class_probability_distribution/np.sum(class_probability_distribution)
    TARGET_AVG_LOSSES = [TARGET_LOSS1,TARGET_LOSS2,TARGET_LOSS3]
    BEST_RUNNING_LOSSES = [np.inf,np.inf] # for LOSS2 AND LOSS 3 ONLY
    BEST_RUNNING_COUNT = 0

    print('class_probability_distribution:',class_probability_distribution)
    print('TARGET_AVG_LOSSES:',TARGET_AVG_LOSSES)
    
    print('\nTarget directories:')
    CKPT_DIR, PROJECT_DIR, MODEL_DIR, LOGGER_DIR, _ = folder_check(FINETUNE_ID, CKPT_DIR='checkpoint')
    IMG_FOLDER = os.path.join(PROJECT_DIR,'imgs') 
    if not os.path.exists(IMG_FOLDER):
        os.mkdir(IMG_FOLDER)
    IMG_DIR = os.path.join(IMG_FOLDER,'result_sample.best.jpeg')
    IMG2_DIR = os.path.join(IMG_FOLDER,'result_sample2.best.jpeg')
    
    if load_from_trove:
        print('Scanning the _treasure_trove to load from:')
        TROVE_DIR, TROVE_PROJECT_DIR, TROVE_MODEL_DIR, TROVE_LOGGER_DIR, _ = folder_check(SOURCE_PROJECT_ID, CKPT_DIR='_treasure_trove')    
        samp, net, logger = setup_main_components(TROVE_MODEL_DIR, TROVE_LOGGER_DIR)
    else:
        samp, net, logger = setup_main_components(MODEL_DIR, LOGGER_DIR)

    DISCOUNT_RATE = 0.95
    lambda1,lambda2 = 1.,100.
    lmon = LossMonitor(average_n_iters=average_n_iters)
    
    ###################################################################
    criterion = nn.CrossEntropyLoss()
    criterion_gc = nn.CrossEntropyLoss()
    criterion_gt = nn.CrossEntropyLoss()
    ###################################################################

    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9,0.9),weight_decay=1e-2)
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.01,0.9),weight_decay=1e-2)

    print('Images will be saved realtime to:\n %s and \n %s'%(str(IMG_DIR),str(IMG2_DIR)))
    start = time.time()
    n_eval, n_correct = None, None
    for ne in range(N_epoch):
        for i in range(n_iter_per_epoch):
            net.train()
            net.zero_grad()

            class_indices = np.random.choice(range(10), batch_size, p=class_probability_distribution)
            x, y0, yg0, ys0 = samp.get_sample_batch(class_indices, device=device)
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

            BEST_RECORD_ACHIEVED = [l1_avg<TARGET_AVG_LOSSES[0],l2_avg<=BEST_RUNNING_LOSSES[0], l3_avg<=BEST_RUNNING_LOSSES[1]]
            BEST_RECORD_ACHIEVED = np.all(BEST_RECORD_ACHIEVED)
            if BEST_RECORD_ACHIEVED:
                BEST_RUNNING_LOSSES = [np.max((l2_avg,DISCOUNT_RATE*BEST_RUNNING_LOSSES[0])),
                    np.max((l3_avg, DISCOUNT_RATE*BEST_RUNNING_LOSSES[1]))] 
                torch.save(net, MODEL_DIR+'.best')

                net.eval()
                save_samples_for_display_and_adjustments(net, samp, device, IMG_DIR, IMG2_DIR)
            
            if realtime_print:
                status = 'NORMAL'
                if AVG_TARGET_ACHIEVED: status = 'AVG_TARGET_ACHIEVED'
                if BEST_RECORD_ACHIEVED: BEST_RUNNING_COUNT += 1
                if i%4==0 or i+1==n_iter_per_epoch:
                    epoch_update = '%s'%('epoch: %s/%s iter:%s/%s'%(str(ne+1),str(N_epoch),str(i+1),str(n_iter_per_epoch)))
                    l1_avg, l2_avg, l3_avg = round(l1_avg,4),round(l2_avg,4),round(l3_avg,4)
                    print('%-96s'%('%s avg losses:%s,%s,%s [%s] lr:%s n_improvement:%s'%(epoch_update,l1_avg, l2_avg, l3_avg, 
                        status, str(optimizer.defaults['lr']), str(BEST_RUNNING_COUNT) )),end='\r')

            if AVG_TARGET_ACHIEVED: break
        if AVG_TARGET_ACHIEVED: break

    end = time.time()
    elapsed = end - start
    print('\n\n')
    if AVG_TARGET_ACHIEVED: print('\n<<AVG_TARGET_ACHIEVED>>\n')

    if BEST_RUNNING_COUNT>0:
        print('Evaluating best model so far.')

        net = torch.load(MODEL_DIR+'.best')
        with torch.no_grad():
            net.eval()
            n_eval, n_correct = 0,0
            for i_eval in range(n_eval_epoch):
                class_indices = np.random.choice(range(10), batch_size)
                x, y0, yg0, ys0 = samp.get_sample_batch(class_indices, device=device)
                y, yg, ys = net(x)
                
                for j_eval, y_pred in enumerate(y):
                    pred_class = int(torch.argmax(y_pred))
                    true_class = int(y0[j_eval])
                    if true_class==pred_class: n_correct+=1
                    n_eval+=1

                if i_eval==0:
                    save_samples_for_display_and_adjustments(net, samp, device, IMG_DIR, IMG2_DIR, print_dir=True)

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

        logger.pickle_data(logger, LOGGER_DIR, tv=(0,0,None), text=None)
    else:
        print('no improvement')

def save_samples_for_display_and_adjustments(net, samp, device, IMG_DIR, IMG2_DIR, print_dir=False):
    class_indices = range(10)
    x, y0, yg0, ys0 = samp.get_sample_batch(class_indices, device=device)
    y, yg, ys = net(x)
    pp.evaluation_visual_results(x, y,yg,ys, y0, yg0, ys0, GPTGenerator=samp.gen, save_dir=IMG_DIR)
    
    # specifically, we want to observe class [8,9], as it has the most mistakes.
    class_indices = np.random.choice([8,9],10)
    x, y0, yg0, ys0 = samp.get_sample_batch(class_indices, device=device)
    y, yg, ys = net(x)
    pp.evaluation_visual_results(x, y,yg,ys, y0, yg0, ys0, GPTGenerator=samp.gen, save_dir=IMG2_DIR)

    if print_dir:
        print('Images saved to:\n %s and \n %s'%(str(IMG_DIR),str(IMG2_DIR)))