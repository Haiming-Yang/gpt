import time, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from .utils import folder_check, Logger, save_ckpt_dir
import src.gpt_mnist.data as ud

from scipy.stats import spearmanr

from .pipeline import compute_attr_one_pixel_target
from .model import ResGPTNet34
from captum.attr import LayerGradCam, Deconvolution, GuidedBackprop # ShapleyValueSampling

import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 0.1 #set the value globally
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def entry(args):
    if args['mode'] == 'basic':
        eval_basic_metrics(args)
    elif args['mode'] == 'sanity_weight_randomization':
        eval_sanity_weight_randomization(args)
    elif args['mode'] == 'compute_AOPC':
        eval_AOPC(args)
    elif args['mode'] == 'view_weight_randomization':
        view_sanity_weight_randomization(args)
    elif args['mode'] == 'view_AOPC':
        view_AOPC(args)
    else:
        raise RuntimeError('Invalid mode specified.')      

        
######################################################
# Evaluation 1. basic metrics
######################################################


def eval_basic_metrics(args):
    print('compute_basic_metrics()')

    PROJECT_ID = args['PROJECT_ID']
    N_EVAL_SAMPLE = args['N_EVAL_SAMPLE']

    CKPT_DIR, PROJECT_DIR, MODEL_DIR, LOGGER_DIR, load_model = folder_check(PROJECT_ID, CKPT_DIR='checkpoint')
    EVAL_DIR = os.path.join(PROJECT_DIR,'eval')
    if not os.path.exists(EVAL_DIR): os.mkdir(EVAL_DIR)

    from .sampler import Pytorch_GPT_MNIST_Sampler
    samp = Pytorch_GPT_MNIST_Sampler(compenv_mode=None, growth_mode=None)

    net = torch.load(MODEL_DIR)
    net = ResGPTNet34(nG0=samp.gen.nG0, Nj=samp.gen.N_neighbor)
    net = torch.load(MODEL_DIR)
    net.output_mode = None
    net.to(device=device)
    net.eval()

    acc = 0
    metrics = {
        'yg_acc':[], 'yg_precision':[],'yg_recall':[],
        'ys_acc':[], 'ys_precision':[],'ys_recall':[],
    }
    with torch.no_grad():
        for i in range(N_EVAL_SAMPLE):
            if (i+1)%10==0 or i+1==N_EVAL_SAMPLE:
                print('%-64s'%('%s/%s'%(str(i+1),str(N_EVAL_SAMPLE))), end='\r')
            x, y0, yg0, ys0 = samp.get_sample_batch_uniform_random(batch_size=1, device=device)
            y, yg, ys = net(x)
            
            # print(y0.shape)
            # print(y.shape)
            y_pred = np.argmax(y[0].clone().detach().cpu().numpy())
            if y_pred==y0.item(): acc+=1
            # raise Exception('debug')

            yg0 = yg0[0].clone().detach().cpu().numpy()
            yg_pred = torch.argmax(yg, dim=1)[0].clone().detach().cpu().numpy()
            sample_yg_acc, sample_yg_precision, sample_yg_recall = compute_one_sample_basic_metrics(yg0.reshape(-1), yg_pred.reshape(-1))
            metrics['yg_acc'].append(sample_yg_acc)
            metrics['yg_precision'].append(sample_yg_precision)
            metrics['yg_recall'].append(sample_yg_recall)

            ys0 = ys0[0].clone().detach().cpu().numpy()
            ys_pred = torch.argmax(ys, dim=1)[0].clone().detach().cpu().numpy()
            sample_ys_acc, sample_ys_precision, sample_ys_recall = compute_one_sample_basic_metrics(ys0.reshape(-1), ys_pred.reshape(-1))
            metrics['ys_acc'].append(sample_ys_acc)
            metrics['ys_precision'].append(sample_ys_precision)
            metrics['ys_recall'].append(sample_ys_recall)
    acc = acc/N_EVAL_SAMPLE
    print('\nacc:%s'%(str(acc)))
    metrics_dir = os.path.join(EVAL_DIR,'basic.csv')
    df = pd.DataFrame(metrics)
    df.to_csv(metrics_dir)

    metrics_summary = {'y_acc':[acc,None]}
    for y in ['yg','ys']:
        for x in ['acc', 'precision', 'recall']:
            metrics_summary['%s_%s'%(y,x)] = [np.mean(metrics['%s_%s'%(y,x)]),np.var(metrics['%s_%s'%(y,x)])**0.5]
    
    print(metrics_summary)
    metrics_summary_dir = os.path.join(EVAL_DIR,'basic_summary.csv')
    dfs = pd.DataFrame(metrics_summary)
    dfs.index=['mean', 'std']
    dfs.to_csv(metrics_summary_dir)
    print('saved to \n 1. %s\n 2. %s'%(str(metrics_dir),str(metrics_summary_dir)))

def compute_one_sample_basic_metrics(y, y0):
    # y, y0 1D arrays of integers 0,1,2...,N, same size
    # if y0 is groundtruth, y[i]=y0[i]=0 is true negative etc.  

    TP_array = ((y>0)*(y0>0)*(y==y0)).astype(int)
    TN_array = ((y==0)*(y0==0)).astype(int)
    FP_array = ((y>0)*(y0==0) + (y>0)*(y0>0)*(y!=y0)).astype(int)
    FN_array = (((y==0)*(y0>0))).astype(int)

    TP = np.sum(TP_array)
    TN = np.sum(TN_array)
    FP = np.sum(FP_array)
    FN = np.sum(FN_array)

    acc = np.mean(y==y0)
    precision = TP/(TP+FP) if TP>0 else 0
    recall = TP/(TP+FN) if TP>0 else 0
    return acc, precision, recall


######################################################
# Evaluation 2. Sanity Check Weight randomization
######################################################

def eval_sanity_weight_randomization(args):
    PROJECT_ID = args['PROJECT_ID']
    N_EVAL_SAMPLE = args['N_EVAL_SAMPLE']
    N_EPOCH = args['N_EPOCH']

    print('N_EVAL_SAMPLE:%s, N_EPOCH:%s'%(str(N_EVAL_SAMPLE),str(N_EPOCH)))

    CKPT_DIR, PROJECT_DIR, MODEL_DIR, LOGGER_DIR, load_model = folder_check(PROJECT_ID, CKPT_DIR='checkpoint')
    EVAL_DIR = os.path.join(PROJECT_DIR,'eval')
    if not os.path.exists(EVAL_DIR): os.mkdir(EVAL_DIR)

    from .sampler import Pytorch_GPT_MNIST_Sampler
    samp = Pytorch_GPT_MNIST_Sampler(compenv_mode=None, growth_mode=None)

    start = time.time()
    for i in range(1,1+5):
        compute_correlations(N_EPOCH, N_EVAL_SAMPLE, MODEL_DIR, samp, cascade_level=i, device=device, save_folder=EVAL_DIR)

    for i in range(1,1+6):
        compute_correlationsGC(N_EPOCH, N_EVAL_SAMPLE, MODEL_DIR, samp,noise_attenuation=0.5, cascade_level=i, 
            target_output='generators', device=device, save_folder=EVAL_DIR)
        compute_correlationsGC(N_EPOCH, N_EVAL_SAMPLE, MODEL_DIR, samp,noise_attenuation=0.5, cascade_level=i, 
            target_output='gen_transformation', device=device, save_folder=EVAL_DIR)
    end = time.time()
    elapsed = end - start
    print('\n\ntime taken %s[s] = %s [min] = %s [hr]'%(str(round(elapsed,1)), str(round(elapsed/60.,1)), str(round(elapsed/3600.,1))))


def prepare_results_dictionary(methods):
    # methods: list of string, each the name of XAI method
    spearman_metrics = {}
    spearman_abs_metrics = {}
    for method in methods: 
        spearman_metrics[method] = []
        spearman_abs_metrics[method] = []
    return spearman_metrics, spearman_abs_metrics

def setup_xai_methods_with_cascaded_randomization(method, net0, net):
    if method=='gradCAM':
        xai0 = LayerGradCam(net0, net0.channel_adj)
        xai = LayerGradCam(net, net.channel_adj)
    elif method == 'deconv':
        xai0 = Deconvolution(net0)
        xai = Deconvolution(net)
    elif method == 'GuidedBP':
        xai0 = GuidedBackprop(net0)
        xai = GuidedBackprop(net)
    else:
        raise RuntimeError('Invalid XAI method.')    
    return xai0, xai

def compute_correlations(N_EPOCH, N_EVAL_SAMPLE, MODEL_DIR, sampler, cascade_level=1, device=None, save_folder=None):
    print('\ncompute_correlations for cascade_level:%s'%(str(cascade_level)))   
    methods = ['gradCAM', 'deconv', 'GuidedBP']
    n_methods = len(methods)
    spearman_metrics, spearman_abs_metrics = prepare_results_dictionary(methods)

    for n_epoch in range(N_EPOCH):
        # different epochs, different weights randomization
        net0 = get_cascade_randomized_net(MODEL_DIR, sampler, output_mode='prediction_only', cascade_level=0, device=device)     
        net = get_cascade_randomized_net(MODEL_DIR, sampler, output_mode='prediction_only', cascade_level=cascade_level, device=device)    

        for j,method in enumerate(methods):
            xai0, xai = setup_xai_methods_with_cascaded_randomization(method, net0, net)
            
            for i in range(N_EVAL_SAMPLE):
                x, y0, yg0, ys0 = sampler.get_sample_batch(class_indices=[np.random.randint(10)], device=device)
                x.requires_grad=True
                
                if method in ['gradCAM','deconv','GuidedBP']:
                    attr0 = xai0.attribute(x.clone(), target=y0).clone().detach().cpu().numpy().reshape(-1)
                    attr = xai.attribute(x.clone(), target=y0).clone().detach().cpu().numpy().reshape(-1)               
                else:
                    raise RuntimeError('Invalid XAI method.')

                attr0_max = np.max(np.abs(attr0))
                if attr0_max>0 and not np.all(np.isnan(attr0)):
                    attr0 = attr0/attr0_max
                else:
                    # yes, this seems to happen. The method may return an array of all zeros.
                    # this will cause spearman to give NaN, which is no good.
                    attr0 = np.random.normal(0,0.01,size=attr.shape)

                attr_max = np.max(np.abs(attr))
                if attr_max>0 and not np.all(np.isnan(attr)):
                    attr = attr/attr_max
                else:
                    attr = np.random.normal(0,0.01,size=attr.shape)

                # print(attr.shape, attr0.shape)
                rho, _ = spearmanr(attr,attr0,axis=None)
                rho_abs, _ = spearmanr(np.abs(attr),np.abs(attr0),axis=None)

                if np.isnan(rho):
                    print(attr)
                    print(attr0)
                    print(np.max(attr), np.min(attr))
                    print(np.max(attr0), np.min(attr0))
                    raise Exception('why?')
                if np.isnan(rho_abs):
                    print(attr)
                    print(attr0)
                    print(np.max(attr), np.min(attr))
                    print(np.max(attr0), np.min(attr0))
                    raise Exception('why?')
                

                spearman_metrics[method].append(rho)
                spearman_abs_metrics[method].append(rho_abs)
                if (i+1)%4==0 or i+1 == N_EVAL_SAMPLE:
                    update_text = 'epoch %s/%s. %s/%s method %s/%s [%s] rho:%s'%(str(n_epoch+1),str(N_EPOCH),str(i+1),str(N_EVAL_SAMPLE),str(j+1),str(n_methods),str(method),str(rho))
                    print('%-64s'%(str(update_text)),end='\r')
    print('\ncomputation done.')

    if save_folder is not None:
        SAVE_DIR1 = os.path.join(save_folder,'y.cascade_rand.%s.csv'%(str(cascade_level)))
        pd.DataFrame(spearman_metrics).to_csv(SAVE_DIR1)
    
        SAVE_DIR2 = os.path.join(save_folder,'abs.y.cascade_rand.%s.csv'%(str(cascade_level)))
        pd.DataFrame(spearman_abs_metrics ).to_csv(SAVE_DIR2)

def get_cascade_randomized_net(MODEL_DIR, sampler, output_mode='prediction_only', cascade_level=1, device=None):
    from .model import ResGPTNet34

    net = ResGPTNet34(nG0=sampler.gen.nG0, Nj=sampler.gen.N_neighbor)
    net = torch.load(MODEL_DIR)
    net.output_mode = output_mode
    net.to(device=device)
    net.eval()

    # print(torch.sum(net.fc.weight.data))

    if cascade_level>=1:
        # print('randomizing fc')
        net.fc.weight.data = net.fc.weight.data*0 + torch.randn(size=(net.fc.weight.data.shape)).to(device=device)

    if cascade_level>=2:
        # print('randomizing layer 4')
        randomizing_one_resnet_sequential_of_basic_blocks(net.backbone.layer4, device=device)

    if cascade_level>=3:
        # print('randomizing layer 3')
        randomizing_one_resnet_sequential_of_basic_blocks(net.backbone.layer3, device=device)

    if cascade_level>=4:
        # print('randomizing layer 2')
        randomizing_one_resnet_sequential_of_basic_blocks(net.backbone.layer2, device=device)

    if cascade_level>=5:
        # print('randomizing layer 1')
        randomizing_one_resnet_sequential_of_basic_blocks(net.backbone.layer1, device=device)

    # print(torch.sum(net.fc.weight.data))
    return net

def compute_correlationsGC(N_EPOCH, N_EVAL_SAMPLE, MODEL_DIR, sampler, target_output='generators', noise_attenuation=1.0,
    cascade_level=1, device=None, save_folder=None):
    # GC means gen configurations
    print('\ncompute_correlationsGC for %s... cascade_level:%s'%(str(target_output),str(cascade_level))) 
    methods = ['gradCAM', 'deconv', 'GuidedBP']
    n_methods = len(methods)
    spearman_metrics, spearman_abs_metrics = prepare_results_dictionary(methods)
    
    for n_epoch in range(N_EPOCH):
        net0 = get_cascade_randomized_net_for_GC(MODEL_DIR, sampler, 
            output_mode=None, cascade_level=0, device=device)
        net = get_cascade_randomized_net_for_GC(MODEL_DIR, sampler, 
            output_mode=None, cascade_level=cascade_level, noise_attenuation=noise_attenuation, device=device)   

        for j,method in enumerate(methods):
            for i in range(N_EVAL_SAMPLE):
                x, y0, yg0, ys0 = sampler.get_sample_batch(class_indices=[np.random.randint(10)], device=device)
                x.requires_grad=True

                y_orig, yg_orig, ys_orig = net0(x) # original unperturbed net
                y, yg, ys = net(x)

                # predicted generaators position before randomization
                gens_pred_orig = yg_orig[0].clone().detach().cpu().numpy()
                gens_pred_orig = np.argmax(gens_pred_orig,axis=0) + 1 # (28,28)      
                if target_output == 'generators':    
                    # predicted generaators position after randomization
                    gens_pred = yg[0].clone().detach().cpu().numpy()
                    gens_pred = np.argmax(gens_pred,axis=0) + 1 # (28,28)    
                elif target_output == 'gen_transformation':
                    # predicted generaators transformations before randomization
                    ss_pred_orig = ys_orig[0].clone().detach().cpu().numpy()
                    ss_pred_orig = np.argmax(ss_pred_orig,axis=0) + 1 # (28,28)                            
                    # predicted generaators transformations after randomization                
                    ss_pred = ys[0].clone().detach().cpu().numpy()
                    ss_pred = np.argmax(ss_pred,axis=0) + 1
                else:
                    raise RuntimeError('Invalid cascade randomization target_output.')

                idxs, idys = np.where(gens_pred_orig>1)
                for idx, idy in zip(idxs,idys):
                    if target_output =='generators':
                        yg_target_pix_orig = torch.argmax(yg_orig,dim=1)[0,idx,idy]    
                        attr0 = compute_attr_one_pixel_target(x, net0, (idx,idy), method, target=yg_target_pix_orig)  
                        attr = compute_attr_one_pixel_target(x, net, (idx,idy), method, target=yg_target_pix_orig)  
                    elif target_output == 'gen_transformation':
                        ys_target_pix_orig = torch.argmax(ys_orig,dim=1)[0,idx,idy]    
                        attr0 = compute_attr_one_pixel_target(x, net0, (idx,idy), method, target=ys_target_pix_orig, wrapper_output='ys_pixel')  
                        attr = compute_attr_one_pixel_target(x, net, (idx,idy), method, target=ys_target_pix_orig, wrapper_output='ys_pixel')

                    attr0_max = np.max(np.abs(attr0))
                    if attr0_max>0:
                        attr0 = attr0/attr0_max
                    else: 
                        # see compute_correlations for similar remark
                        attr0 = np.random.normal(0,0.1,size=attr0.shape)

                    attr_max = np.max(np.abs(attr))
                    if attr_max>0:
                        attr = attr/attr_max
                    else:
                        attr = np.random.normal(0,0.01,size=attr.shape)

                    rho, _ = spearmanr(attr,attr0,axis=None)
                    rho_abs, _ = spearmanr(np.abs(attr),np.abs(attr0),axis=None)

                    if np.isnan(rho):
                        print(attr)
                        print(attr0)
                        print(np.max(attr), np.min(attr))
                        print(np.max(attr0), np.min(attr0))
                        raise Exception('why?')
                    if np.isnan(rho_abs):
                        print(attr)
                        print(attr0)
                        print(np.max(attr), np.min(attr))
                        print(np.max(attr0), np.min(attr0))
                        raise Exception('why?')
                    spearman_metrics[method].append(rho)
                    spearman_abs_metrics[method].append(rho_abs)
                if (i+1)%4==0 or i+1 == N_EVAL_SAMPLE:
                    update_text = 'epoch:%s/%s. %s/%s method %s/%s [%s] rho:%s'%(str(n_epoch+1),str(N_EPOCH),str(i+1),str(N_EVAL_SAMPLE),str(j+1),str(n_methods),str(method),str(rho))
                    print('%-64s'%(str(update_text)),end='\r')                    


    if save_folder is not None:
        if target_output =='generators': filetag = 'yg'
        elif target_output=='gen_transformation': filetag = 'ys'
        
        SAVE_DIR1 = os.path.join(save_folder,'%s.cascade_rand.%s.csv'%(str(filetag),str(cascade_level)))
        pd.DataFrame(pad_dict_for_dataframe(spearman_metrics, padding= -999)).to_csv(SAVE_DIR1)
    
        SAVE_DIR2 = os.path.join(save_folder,'abs.%s.cascade_rand.%s.csv'%(str(filetag),str(cascade_level)))
        pd.DataFrame(pad_dict_for_dataframe(spearman_abs_metrics, padding= -999) ).to_csv(SAVE_DIR2)


def pad_dict_for_dataframe(d, padding= -999):
    max_length = 0
    for xkey, x in d.items():
        if len(x)>max_length: max_length=len(x)

    for xkey, x in d.items():
        this_arr = np.zeros(max_length) + padding
        this_arr[:len(x)] = x
        d[xkey] = this_arr
    return d

def get_uniform_random_initialization(this_shape, max_abs_value, noise_attenuation=1. ,device=None):
    return max_abs_value * (2*torch.rand(size=this_shape)-1).to(device=device) * noise_attenuation

def get_cascade_randomized_net_for_GC(MODEL_DIR, sampler, output_mode='prediction_only', 
    cascade_level=1, noise_attenuation=0.5, device=None):
    from .model import ResGPTNet34

    net = ResGPTNet34(nG0=sampler.gen.nG0, Nj=sampler.gen.N_neighbor)
    net = torch.load(MODEL_DIR)
    net.output_mode = output_mode
    net.to(device=device)
    net.eval()

    # print(torch.sum(net.cyg.c1.weight.data))

    if cascade_level>=1:
        # print('randomizing output layer')
        s = net.cyg.c1.weight.data.shape
        max_val = torch.max(net.cyg.c1.weight.data)
        net.cyg.c1.weight.data = get_uniform_random_initialization(s, max_val, noise_attenuation=noise_attenuation, device=device)
        s = net.cyg.c2.weight.data.shape
        max_val = torch.max(net.cyg.c2.weight.data)
        net.cyg.c2.weight.data = get_uniform_random_initialization(s, max_val, noise_attenuation=noise_attenuation, device=device)

        s = net.cys.c1.weight.data.shape
        max_val = torch.max(net.cys.c1.weight.data)
        net.cys.c1.weight.data = get_uniform_random_initialization(s, max_val, noise_attenuation=noise_attenuation, device=device)
        s = net.cys.c2.weight.data.shape
        max_val = torch.max(net.cys.c2.weight.data)
        net.cys.c2.weight.data = get_uniform_random_initialization(s, max_val, noise_attenuation=noise_attenuation, device=device)
        
    if cascade_level>=2:
        # print('randomizing eb2')
        s = net.eb2.c1.weight.data.shape
        max_val = torch.max(net.eb2.c1.weight.data) 
        net.eb2.c1.weight.data = get_uniform_random_initialization(s, max_val, noise_attenuation=noise_attenuation, device=device)
        s = net.eb2.bn1.weight.data.shape
        max_val = torch.max(net.eb2.bn1.weight.data)
        net.eb2.bn1.weight.data = get_uniform_random_initialization(s, max_val,noise_attenuation=noise_attenuation, device=device)

    if cascade_level>=3:
        # print('randomizing eb1')
        s = net.eb1.c1.weight.data.shape
        max_val = torch.max(net.eb1.c1.weight.data) 
        net.eb1.c1.weight.data = get_uniform_random_initialization(s, max_val, noise_attenuation=noise_attenuation, device=device)
        s = net.eb1.bn1.weight.data.shape
        max_val = torch.max(net.eb1.bn1.weight.data) 
        net.eb1.bn1.weight.data = get_uniform_random_initialization(s, max_val,noise_attenuation=noise_attenuation, device=device)

    if cascade_level>=4:
        # print('randomizing eb0')
        s = net.eb0.c1.weight.data.shape
        max_val = torch.max(net.eb0.c1.weight.data) 
        net.eb0.c1.weight.data = get_uniform_random_initialization(s, max_val, noise_attenuation=noise_attenuation, device=device)
        s = net.eb0.bn1.weight.data.shape
        max_val = torch.max(net.eb0.bn1.weight.data) 
        net.eb0.bn1.weight.data = get_uniform_random_initialization(s, max_val,noise_attenuation=noise_attenuation, device=device)
        
    if cascade_level>=5:
        # print('randomizing resnet layer 2')
        randomizing_one_resnet_sequential_of_basic_blocks(net.backbone.layer2, device=device)

    if cascade_level>=6:
        # print('randomizing resnet layer 1')
        randomizing_one_resnet_sequential_of_basic_blocks(net.backbone.layer1, device=device)

    net.to(device=device)
    return net    

def randomizing_one_resnet_sequential_of_basic_blocks(layer, device=None):
    """
    Let resnet34 be net.
    There are 4 sequential layers in net, each containing different number of BasicBlock
    
    Example usage:
    for i in [1,2,3,4]:
        print(torch.sum(getattr(getattr(net.backbone,'layer%s'%(str(i))),'0') .conv1.weight.data))
        randomizing_one_resnet_sequential_of_basic_blocks(getattr(net.backbone,'layer%s'%(str(i))),
            device=device)
        print(torch.sum(getattr(getattr(net.backbone,'layer%s'%(str(i))),'0') .conv1.weight.data)) # check that values change
    """
    n_basic_block = np.sum([1 for x in layer.children()])
    for j in range(n_basic_block):
        randomizing_one_resnet_basic_block(getattr(layer,str(j)) ,device=device)

def randomizing_one_resnet_basic_block(bblock, sigma=0.1, device=None, debug=0):
    # example
    # randomizing_one_resnet_basic_block(getattr(net.backbone.layer1,'0'), device=device)
    for x in ['conv1','bn1','conv2','bn2']:
        s = getattr(bblock,x).weight.data.shape 
        getattr(bblock,x).weight.data = getattr(bblock,x).weight.data * 0 + (1-debug) * sigma* torch.randn(size=s).to(device=device)
        if getattr(bblock,x).bias is not None:
            sb = getattr(bblock,x).bias.data.shape 
            getattr(bblock,x).bias.data = getattr(bblock,x).bias.data * 0 + (1-debug) * sigma* torch.randn(size=sb).to(device=device)


######################################################
# Evaluation 3. MoRF and AOPC
######################################################

def eval_AOPC(args):
    print('Evaluation by AOPC')
    PROJECT_ID = args['PROJECT_ID']
    N_EVAL_SAMPLE = args['N_EVAL_SAMPLE']

    CKPT_DIR, PROJECT_DIR, MODEL_DIR, LOGGER_DIR, load_model = folder_check(PROJECT_ID, CKPT_DIR='checkpoint')
    EVAL_DIR = os.path.join(PROJECT_DIR,'eval')
    if not os.path.exists(EVAL_DIR): os.mkdir(EVAL_DIR)

    from .sampler import Pytorch_GPT_MNIST_Sampler
    samp = Pytorch_GPT_MNIST_Sampler(compenv_mode=None, growth_mode=None)

    net = ResGPTNet34(nG0=samp.gen.nG0, Nj=samp.gen.N_neighbor)
    net = torch.load(MODEL_DIR)
    net.output_mode = None
    net.to(device=device)
    net.eval()

    perturbation_steps = np.array(range(1,1+75)) # 75 steps perturbation
    methods = ['gradCAM', 'deconv', 'GuidedBP']
    AOPC_dict, AOPC_abs_dict = setup_AOPC_dicts(perturbation_steps, methods)

    start = time.time()
    for method in methods:
        AOPC = compute_AOPC_for_y_pred(perturbation_steps, method, net, samp, N_EVAL_SAMPLE, abs_attr=False)
        AOPC_dict[method] = [aopc for L, aopc in AOPC.items()]

        AOPC_abs = compute_AOPC_for_y_pred(perturbation_steps, method, net, samp, N_EVAL_SAMPLE, abs_attr=True)
        AOPC_abs_dict[method] = [aopc for L, aopc in AOPC_abs.items()]

    save_folder = EVAL_DIR
    SAVE_DIR1 = os.path.join(save_folder,'y.AOPC.csv')
    pd.DataFrame(AOPC_dict).to_csv(SAVE_DIR1)

    SAVE_DIR2 = os.path.join(save_folder,'abs.y.AOPC.csv')
    pd.DataFrame(AOPC_abs_dict ).to_csv(SAVE_DIR2)

    end = time.time()
    elapsed = end - start
    print('\n\ntime taken %s[s] = %s [min] = %s [hr]'%(str(round(elapsed,1)), str(round(elapsed/60.,1)), str(round(elapsed/3600.,1))))


def setup_AOPC_dicts(perturbation_steps, methods):
    AOPC_dict = {}
    AOPC_dict['perturbation_steps'] = perturbation_steps

    AOPC_abs_dict = {}
    AOPC_abs_dict['perturbation_steps'] = perturbation_steps

    return AOPC_dict, AOPC_abs_dict

def compute_AOPC_for_y_pred(perturbation_steps, method, net, sampler, N_EVAL_SAMPLE, abs_attr=False):
    AOPC = {}
    for L in perturbation_steps: 
        AOPC[L] = []

    x, y0, yg0, ys0 = sampler.get_sample_batch_uniform_random(batch_size=1, device=device)
    net.output_mode = None
    y, yg, ys = net(x)
    print('compute_AOPC_for_y_pred... abs value?%s'%(str(abs_attr)))
    for i in range(N_EVAL_SAMPLE):
        update_text = '%s/%s samples [%s]'%(str(i+1),str(N_EVAL_SAMPLE),str(method))
        print('%-64s'%(update_text), end='\r')
        sample_AOPC_over_all_perturbations = compute_one_sample_AOPC_for_y_pred(perturbation_steps, x, y, y0, method, net, abs_attr=abs_attr)
        for L in perturbation_steps: 
            AOPC[L].append(sample_AOPC_over_all_perturbations[L])
    
    for L in perturbation_steps: 
        AOPC[L] = np.mean(AOPC[L])     
    print('\nall samples done')
    return AOPC

def compute_one_sample_AOPC_for_y_pred(perturbation_steps, x, y, y0, method, net, abs_attr=False):
    net.output_mode = 'prediction_only'
    xai = setup_xai_method(method, net)
    L_MAX = perturbation_steps[-1]

    if method in ['gradCAM','deconv','GuidedBP']:
        attr = xai.attribute(x.clone(), target=y0).clone().detach().cpu().numpy()               
    else:
        raise RuntimeError('Invalid XAI method.')

    if abs_attr:
        attr = np.abs(attr)

    attr_max = np.max(np.abs(attr))
    if attr_max>0:
        attr = attr/attr_max

    x_max = np.max(np.abs(x.clone().detach().cpu().numpy()))
    sample_sum_of_diffs = {}
    running_sample_sum_of_diff = 0
    yc = y[0,y0[0]]
    for L in range(1,L_MAX +1):
        max_index = np.where(attr==np.max(attr),1,0)
        # cgoose one out of the ranoom max values
        # in case heatmaps contain multiple pixels with the same max values
        rindex = max_index + np.random.uniform(0,1,size=max_index.shape)
        rindex = np.where(rindex==np.max(rindex),1,0)

        attr = (1-rindex)*attr + rindex*0

        rindex = torch.tensor(rindex).to(device=device)        
        x_perturbed = (1-rindex)*x+ rindex * x_max * np.random.uniform(-0.5,0.5)
        y_c_perturbed = net(x_perturbed)[0,y0[0]]
        running_sample_sum_of_diff += (yc.item() - y_c_perturbed.item())

        if L in perturbation_steps:
            sample_sum_of_diffs[L] = running_sample_sum_of_diff/(L+1)
        # print(yc.item(), y_c_perturbed.item())
    sample_AOPC_over_all_perturbations = sample_sum_of_diffs
    return sample_AOPC_over_all_perturbations        

def setup_xai_method(method, net):
    if method=='gradCAM':
        xai = LayerGradCam(net, net.channel_adj)
    elif method == 'deconv':
        xai = Deconvolution(net)
    elif method == 'GuidedBP':
        xai = GuidedBackprop(net)
    else:
        raise RuntimeError('Invalid XAI method.')    
    return xai


##########################################
# 4. Viewing results
##########################################

def view_AOPC(args):
    print('View AOPC')
    PROJECT_ID = args['PROJECT_ID']

    CKPT_DIR, PROJECT_DIR, MODEL_DIR, LOGGER_DIR, load_model = folder_check(PROJECT_ID, CKPT_DIR='checkpoint')
    EVAL_DIR = os.path.join(PROJECT_DIR,'eval')
    methods = ['gradCAM', 'deconv','GuidedBP']
    save_folder = EVAL_DIR

    plt.figure()

    SAVE_DIR1 = os.path.join(save_folder,'y.AOPC.csv')

    df = pd.read_csv(SAVE_DIR1)
    L = np.array(df['perturbation_steps'])
    plt.gcf().add_subplot(211)
    for method in methods:
        method_aopc = np.array(df[method])
        plt.gca().plot(L,method_aopc, label=method, linewidth=0.5)
    plt.gca().set_ylabel('AOPC no ABS')

    SAVE_DIR2 = os.path.join(save_folder,'abs.y.AOPC.csv')
    df2 = pd.read_csv(SAVE_DIR2)
    L = np.array(df2['perturbation_steps'])
    plt.gcf().add_subplot(212)
    for method in methods:
        method_aopc = np.array(df2[method])
        plt.gca().plot(L,method_aopc, label=method, linewidth=0.5)
    plt.gca().set_ylabel('AOPC with ABS')
    plt.gca().set_xlabel('no of perturbation steps')
    plt.legend()
    
    plt.show()

def filter_spearman_values(spearman_value, dummy_value=-999):
    filtered_spr = []
    n_nan, n_total = 0, 0
    for x in spearman_value:
        x = np.float(x)
        if x==-999: continue
        n_total+=1
        if np.isnan(x): 
            n_nan+=1;  continue
        filtered_spr.append(x)    
    fraction_nan = n_nan/n_total
    return filtered_spr, n_nan, fraction_nan

def view_sanity_weight_randomization(args):
    print('view_sanity_weight_randomization')

    PROJECT_ID = args['PROJECT_ID']

    CKPT_DIR, PROJECT_DIR, MODEL_DIR, LOGGER_DIR, load_model = folder_check(PROJECT_ID, CKPT_DIR='checkpoint')
    EVAL_DIR = os.path.join(PROJECT_DIR,'eval')
    methods = ['gradCAM', 'deconv','GuidedBP']
    save_folder = EVAL_DIR

    # y
    def plot_sanity_check_cascade_randomization(X,EVAL_DIR, methods, output_tag='y', abs_attr=False, set_legend=False,
        plot_title=None, off_yticks=False, off_xticks=False, xticks_labels=None, ylabel=None):
        print('plot_sanity_check_cascade_randomization for %s, abs?:%s'%(output_tag, str(abs_attr)))
        spearman = {}
        sprman_info = {}
        for method in methods:
            spearman[method] = []
            sprman_info[method] = {}
            for cascade_no in X:
                sprman_info[method][cascade_no] = {}

                if abs_attr:
                    save_dir = os.path.join(EVAL_DIR, 'abs.' + '%s.cascade_rand.%s.csv'%(str(output_tag),str(cascade_no)))
                else:
                    save_dir = os.path.join(EVAL_DIR, '%s.cascade_rand.%s.csv'%(str(output_tag),str(cascade_no)))

                df = pd.read_csv(save_dir)
                spr = np.array(df[method])
                filtered_spr, n_nan, fraction_nan = filter_spearman_values(spr)
                spr_mean = np.mean(filtered_spr)

                spearman[method].append(spr_mean)
                sprman_info[method][cascade_no]['nan n and fraction'] = (n_nan, fraction_nan)
                # print('[%s]'%(str(method)),'cascade_no:',cascade_no, 'n_nan:', n_nan , '% nan:', round(n_nan/n_total,3))
        # print(spearman)
        
        for method in spearman:
            print('plotting spearman results for %s'%(str(method)))
            plt.gca().plot(X,spearman[method], label=method, marker='.', linewidth=0.5)

            plt.gca().set_ylim([-1.1,1.1])
            if off_yticks:
                plt.gca().set_yticks([])
            
            if off_xticks:
                plt.gca().set_xticks([])
            else:
                if xticks_labels is None:
                    xticks_labels = X
                print(X)
                plt.gca().set_xticks(X)
                plt.gca().set_xticklabels(xticks_labels, rotation=-60)

            if not ylabel is None: plt.gca().set_ylabel(ylabel)
            if not plot_title is None:
                plt.gca().set_title(plot_title)

            for cascade_no in sprman_info[method]:
                print('  cascade_no %s: %s'%(str(cascade_no),str(sprman_info[method][cascade_no])))

        if set_legend:
            plt.legend()
        plt.tight_layout()

    plt.figure()
    plt.gcf().add_subplot(231)
    plot_sanity_check_cascade_randomization([1,2,3,4,5], EVAL_DIR, methods, 
        output_tag='y', abs_attr=False, plot_title='y', off_xticks=True, ylabel='Rank correlation\nno ABS')
    plt.gcf().add_subplot(232)
    plot_sanity_check_cascade_randomization([1,2,3,4,5,6], EVAL_DIR, methods, 
        output_tag='yg', abs_attr=False, plot_title='yg', off_xticks=True, off_yticks=True)
    plt.gcf().add_subplot(233)
    plot_sanity_check_cascade_randomization([1,2,3,4,5,6], EVAL_DIR, methods, 
        output_tag='ys', abs_attr=False, plot_title='ys', off_xticks=True, off_yticks=True)

    plt.gcf().add_subplot(234)
    plot_sanity_check_cascade_randomization([1,2,3,4,5] ,EVAL_DIR, methods, 
        output_tag='y', abs_attr=True, set_legend=True, ylabel='Rank correlation\nABS', 
        xticks_labels=['ResNet $_{fc}$', 'ResNet $_{L4}$', 'ResNet $_{L3}$', 'ResNet $_{L2}$','ResNet $_{L1}$'])
    plt.gcf().add_subplot(235)
    plot_sanity_check_cascade_randomization([1,2,3,4,5,6] ,EVAL_DIR, methods, 
        output_tag='yg', abs_attr=True, off_yticks=True,
                xticks_labels=['cyg', 'EB2', 'EB1', 'EB0', 'ResNet $_{2}$','ResNet $_{1}$'])
    plt.gcf().add_subplot(236)
    plot_sanity_check_cascade_randomization([1,2,3,4,5,6] ,EVAL_DIR, methods, 
        output_tag='ys', abs_attr=True,  off_yticks=True,
                xticks_labels=['cys', 'EB2', 'EB1', 'EB0', 'ResNet $_{2}$','ResNet $_{1}$'])
    plt.show()