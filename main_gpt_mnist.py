import argparse, os
from src.gpt_mnist.pipeline import entry

if __name__=='__main__':
    DESC = """Welcome!
    We use ideas from general pattern theory for explainable AI analysis.
    This script handles codes src/gpt_mnist folder.

    Note:
    Bool String: Bool, but parsed as string, so input either 0 or 1.
    """
    MODES_DESC = """Available modes: training (default), view_losses, generate_samples, finetune, heatmaps, heatmapsGC, gen_dist
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

    # Main options
    parser.add_argument('--PROJECT_ID', default='Simple0001', type=str, help=None)
    parser.add_argument('--mode', default='training', type=str, help=MODES_DESC)
    parser.add_argument('--N_EPOCH', default=1, type=int, help=None)
    parser.add_argument('--N_PER_EPOCH', default=16, type=int, help='No. of batches per epochs')
    parser.add_argument('--batch_size', default=16, type=int, help=None)
    parser.add_argument('--track_every_epoch',default=1,type=str,help='Bool String')
    parser.add_argument('--regularizations',default=None, help='None (default), output_size')
    parser.add_argument('--n_eval_epoch', default=10, type=int)
    parser.add_argument('--n_eval_batch', default=10, type=int)

    # Peripheral settings
    parser.add_argument('--realtime_print',default=0,type=str, help='Bool String')
    parser.add_argument('--debug_target',default=0,type=str, help='Bool String')

    # View settings
    parser.add_argument('--random_batch', default=1,type=str, help='Bool String')
    parser.add_argument('--iter0_n_loss', default=None, type=int, help=None)
    parser.add_argument('--average_n_loss', default=None, type=int, help=None)

    # Fine tuning options
    parser.add_argument('--FINETUNE_ID', default='FINETUNE0001', type=str, help=None)
    parser.add_argument('--load_from_trove',default=1,type=str, help='Bool String')
    parser.add_argument('--TARGET_LOSS1', default=1e-5, type=float, help=None)
    parser.add_argument('--TARGET_LOSS2', default=0.0002, type=float, help=None)
    parser.add_argument('--TARGET_LOSS3', default=0.0002, type=float, help=None)
    parser.add_argument('--learning_rate', default=1e-5, type=float, help=None)
    
    args = vars(parser.parse_args())    
    # print(args)

    ROOT_DIR_MODE = args['ROOT_DIR_MODE']
    ROOT_DIR = args['ROOT_DIR']
    if ROOT_DIR_MODE =='rel':
        os.chdir(os.path.join(os.getcwd(),ROOT_DIR))
    elif ROOT_DIR_MODE == 'abs':
        os.chdir(ROOT_DIR)

    entry(args)

