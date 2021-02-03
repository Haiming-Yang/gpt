

def redirect(args):
    print('redirecting to specificing finetuner...')

    PROJECT_ID = args['PROJECT_ID']
    if PROJECT_ID == 'NSCC_NICE1':
        from .finetune_NSCC_NICE1 import do_finetune
        do_finetune(args)
        return
    if PROJECT_ID == 'NSCC_NICE2':
        from .finetune_NSCC_NICE2 import do_finetune
        do_finetune(args)
        return
    if PROJECT_ID == 'Simple0001':
        from .finetune_Simple0001 import do_finetune
        do_finetune(args)
        return
    print('Cannot find PROJECT_ID: %s in _treasure_trove. '%(PROJECT_ID))