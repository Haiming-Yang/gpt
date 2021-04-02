import argparse, os
from src.gpt_genmod.pipeline import entry

if __name__=='__main__':
    DESC = """Welcome!
    """

    MODES_DESC = """Available modes: info (default), training
    """

    DIR_HELP = """ If run from gpt folder, set to None.\n
    If run from another folder, use:\n
    1. --ROOT_DIR_MODE abs, if you supply absolute dir to --ROOT_DIR
    2. --ROOT_DIR_MODE rel, if you supply relative dir to --ROOT_DIR
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description=DESC)
    parser.add_argument('--ROOT_DIR', default=None,help=DIR_HELP)
    parser.add_argument('--ROOT_DIR_MODE', default=None,help=None)
    parser.add_argument('--DATA_DIR', default='data',help=None)

    parser.add_argument('--PROJECT_ID', default='genPROJECT0001', type=str, help=None)
    parser.add_argument('--mode', default='info', type=str, help=MODES_DESC)
    parser.add_argument('--dataset', default='cifar10', type=str, help=None)
    parser.add_argument('--model', default='SimpleGrowth', type=str, help="SimpleGrowth (default)")
    parser.add_argument('--n', default=1, type=int, help=None)

    # training
    parser.add_argument('--learning_rate', default=0.001, type=float, help=None)
    parser.add_argument('--N_EPOCH', default=1, type=int, help=None)
    parser.add_argument('--batch_size', default=16, type=int, help=None)
    parser.add_argument('--AVG_LOSS_EVERY_N_ITER', default=12, type=int, help=None)    

    # sample
    parser.add_argument('--sampling_mode', default=None, help=None)    

    # utils
    parser.add_argument('--SAVE_IMG_EVERY_N_ITER', default=240, type=int, help=None)
    parser.add_argument('--STOP_AT_N_ITER', default=16, type=int, help=None)
    parser.add_argument('--realtime_print', default='0', type=str, help="Bool as string, either 1 or 0.")
    parser.add_argument('--debug_mode', default='0', type=str, help="Bool as string, either 1 or 0.")

    # convenience
    parser.add_argument('--EVALUATE', default='1', type=str, help="Bool as string, either 1 or 0.")
    parser.add_argument('--OBSERVE', default='0', type=str, help="Bool as string, either 1 or 0.")



    args = vars(parser.parse_args())    
    # print(args)

    ROOT_DIR_MODE = args['ROOT_DIR_MODE']
    ROOT_DIR = args['ROOT_DIR']

    if ROOT_DIR_MODE =='rel':
        os.chdir(os.path.join(os.getcwd(),ROOT_DIR))
    elif ROOT_DIR_MODE == 'abs':
        os.chdir(ROOT_DIR)

    if args['mode'] == 'info': 
        print(MODES_DESC); exit()
    entry(args)