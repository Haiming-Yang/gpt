import argparse, os
from src.gpt_benas.pipeline import entry

if __name__=='__main__':
    DESC = """Welcome!
    """

    MODES_DESC = """Available modes: training (default), example_benas_conv, example_benas_deconv, example_model"""

    DIR_HELP = """ If run from gpt folder, set to None.\n
    If run from another folder, use:\n
    1. --ROOT_DIR_MODE abs, if you supply absolute dir to --ROOT_DIR
    2. --ROOT_DIR_MODE rel, if you supply relative dir to --ROOT_DIR
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description=DESC)
    parser.add_argument('--ROOT_DIR', default=None,help=DIR_HELP)
    parser.add_argument('--ROOT_DIR_MODE', default=None,help=None)

    parser.add_argument('--mode', default='training', type=str, help=MODES_DESC)

    args = vars(parser.parse_args())    
    # print(args)

    ROOT_DIR_MODE = args['ROOT_DIR_MODE']
    ROOT_DIR = args['ROOT_DIR']
    if ROOT_DIR_MODE =='rel':
        os.chdir(os.path.join(os.getcwd(),ROOT_DIR))
    elif ROOT_DIR_MODE == 'abs':
        os.chdir(ROOT_DIR)

    entry(args)
