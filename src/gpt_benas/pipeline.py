import torch

def entry(args):
    print(args)

    if args['mode'] == 'training':
        training(args)
    elif args['mode'] == 'example_benas_conv':
        run_example_benas_conv(args)
    elif args['mode'] == 'example_benas_deconv':
        run_example_benas_deconv(args)
    elif args['mode'] == 'example_model':
        run_example_net(args)

def training(args):
    print('training mode')

    raise Exception('TBD')

def run_example_net(args):
    print('run test model')

    from src.gpt_benas.model import AutoGPTNet, get_dummy_dag2
    
    x = torch.rand(size=(4,1,28,28))
    prev_nodes, modules, downsampler = get_dummy_dag2(mode='dag_parts')
    upsampler = 'us2'

    # for now set all to have the same dag
    dags = {
        'cb1': {'prev_nodes':prev_nodes, 'modules' : modules, 'downsampler': downsampler},
        'cb2': {'prev_nodes':prev_nodes, 'modules' : modules, 'downsampler': downsampler},
        'cb3': {'prev_nodes':prev_nodes, 'modules' : modules, 'downsampler': downsampler},
        'db3': {'prev_nodes':prev_nodes, 'modules' : modules, 'upsampler': upsampler},
        'db2': {'prev_nodes':prev_nodes, 'modules' : modules, 'upsampler': upsampler},
        'db1': {'prev_nodes':prev_nodes, 'modules' : modules, 'upsampler': upsampler},
    }

    net = AutoGPTNet()
    x, yg, ys = net.forward(x, dags, verbose=100)



def run_example_benas_conv(args):
    from src.gpt_benas.model import BENASConvBlock, get_dummy_dag, get_dummy_dag2    
    for i in [1,2,3]:
        print('\n\n============ example %s ============'%(str(i)))
        if i ==1:
            num_blocks = 5
            f = get_dummy_dag
        elif i == 2:
            num_blocks = 12
            f = get_dummy_dag2
        elif i==3:
            print('This example will throw error due to wrong number of blocks specified.')
            num_blocks = 5
            f = get_dummy_dag2

        x = torch.rand(size=(4,16,28,28))
        prev_nodes, modules, downsampler = f(mode='dag_parts')
        
        bblock = BENASConvBlock(input_channel=16, num_blocks=num_blocks)
        try:
            dag = bblock.construct_dag(prev_nodes, modules, appendage=downsampler)
            print('construct_dag done...')
            for d,d1 in dag.items():
                print(d,d1)

            y, y_downsampled = bblock.forward(x, num_blocks, dag, verbose=100)
            print('y.shape:',y.shape)
            print('y_downsampled.shape:',y_downsampled.shape)
        except:
            print('error demonstrated!')

def run_example_benas_deconv(args):
    from src.gpt_benas.model import BENASDeConvBlock, get_dummy_dag, get_dummy_dag2    
    for i in [1,2]:
        print('\n\n============ example %s ============'%(str(i)))
        if i ==1:
            num_blocks = 5
            f = get_dummy_dag
        elif i == 2:
            num_blocks = 12
            f = get_dummy_dag2

        x = torch.rand(size=(4,16,28,28))
        prev_nodes, modules, upsampler = f(mode='dag_parts', appendage='us2')
        
        bblock = BENASDeConvBlock(input_channel=16, num_blocks=num_blocks)

        dag = bblock.construct_dag(prev_nodes, modules, appendage=upsampler)
        print('construct_dag done...')
        for d,d1 in dag.items():
            print(d,d1)

        y, y_upsampler = bblock.forward(x, num_blocks, dag, verbose=100)
        print('y.shape:',y.shape)
        print('y_upsampler.shape:',y_upsampler.shape)