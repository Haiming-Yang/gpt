from .data import *
import os, pickle
from src.utils.utils import FastPickleClient

def folder_check(PROJECT_ID, CKPT_DIR=None):
    if CKPT_DIR is None:
        CKPT_DIR = 'checkpoint'
    PROJECT_DIR = os.path.join(CKPT_DIR, PROJECT_ID)
    MODEL_DIR = os.path.join(PROJECT_DIR, 'model.net')
    LOGGER_DIR = os.path.join(PROJECT_DIR, 'logger.data')

    if not os.path.exists(CKPT_DIR):
        os.mkdir(CKPT_DIR)
    if not os.path.exists(PROJECT_DIR):
        os.mkdir(PROJECT_DIR)
    load_model = os.path.exists(MODEL_DIR)
    print('CKPT_DIR    : %s\nPROJECT_DIR : %s\nMODEL_DIR   : %s'%(str(CKPT_DIR),str(PROJECT_DIR),str(MODEL_DIR)))
    
    return CKPT_DIR, PROJECT_DIR, MODEL_DIR, LOGGER_DIR, load_model


class Logger(FastPickleClient):
    def __init__(self, ):
        super(Logger, self).__init__()
        self.save_text = 'Logger saving data...'
        self.load_text = 'Logger loading data...'

        self.iter_array = []
        self.loss_array = {'class_pred_loss':[],'gen_config_loss':[],'gen_transform_loss':[]}
        self.n_th_run = 0
        self.records = {
            # k-th run: {
            #   'time': time_in_secs, 
            #   'N_epoch': N_EPOCH,
            #   'n_iter_per_epoch': N_PER_EPOCH,
            #   'batch_size':batch_size,
            # }
            # (k+1)th run: {...}
            # ...
            # n-th run: {...}
            'by_iter':{
            #   iter_1000: {'n_correct': n_correct, 'n_eval':n_eval,},
            #   iter_2000: {'n_correct': n_correct, 'n_eval':n_eval,},
            #   ....            
            }
        }


def save_ckpt_dir(PROJECT_DIR, model):
    this_iter = int(1e8+model.tracker['iter'])

    IMG_FOLDER = os.path.join(PROJECT_DIR,'imgs') 
    if not os.path.exists(IMG_FOLDER):
        os.mkdir(IMG_FOLDER)
    IMG_DIR = os.path.join(IMG_FOLDER,'result_sample.%s.jpeg'%(str(this_iter)[1:]))
    MODEL_DIR = os.path.join(PROJECT_DIR, 'model.%s.net'%(str(this_iter)[1:]))
    return IMG_DIR, MODEL_DIR