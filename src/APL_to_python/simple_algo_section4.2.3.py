import numpy as np
import argparse
from simple_algo_utils import *


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,description=None)
    parser.add_argument('--example_n', default=1, type=int, help=None)
    parser.add_argument('--verbose', default=0, type=int, help=None)
    args = vars(parser.parse_args())    

    example_n = args['example_n']
    verbose = args['verbose']
    AVAILABLE_EXAMPLES = [1,2]
    SECTION_LABELS = [None,1,1]
    PAGES = [None, 223, 225]
    T_MAX_LIST = [None,3,9]

    try:
        print('Example from Ulf Grenander\' textbook section 4.2.3.%s page %s'%(
            str(SECTION_LABELS[example_n]),str(PAGES[example_n])))
        print('example_n : %s'%(str(example_n)))
    except:
        print('EXAMPLE NOT FOUND.')
        print('The available example numbers for --example_n are %s'%(str(AVAILABLE_EXAMPLES)))
        exit()

    print("==== SETTOPOLOGY ====")
    Nj = 4 # no. of neighbors
    J = np.array([ [1,0], [0,1], [-1,0], [0,-1]]) # (X,Y) coordinates of neighbors
    assert(J.shape[0]==Nj)
    print(J, '\n')


    print("==== SETBSG ====")
    # the group used was \mathbb{Z}_4
    if example_n in [1]:
        L = 10
    elif example_n in [2]:
        L = 20

    nBSG = 4
    BSG = setBSG(nBSG, Nj)
    print('BSG:\n',BSG,'\n')


    print("==== SETG0 ====")
    age = np.zeros(shape=(L,L))
    if example_n in [1]:
        G0 = np.array([[0,0,0,0],[3,3,3,3],[0,0,3,3],[0,0,0,0]])
    elif example_n in [2]:
        G0 = [[0,0,0,0], [2,0,0,0],]
        for i in range(3,1+9): G0.append([i,0,i,0])
        G0.append([0,0,10,0])
        G0 = np.array(G0)

    nG0 = G0.shape[0]-1 # no of generators, excluding emtpy generator
    alpha = range(nG0+1) 
    print('G0:\n',G0,'\n')
    print('alpha:\n',alpha,'\n')

    print("==== SETEXTG0 =====")
    GE = setEXTG0(nBSG, G0, Nj, BSG)
    print('GE:\n',GE,'\n')


    print("==== SETCINIT ====")
    CE = np.ones(shape=(L,L))
    if example_n in [1]:
        CE[4,4] = 5
    elif example_n in [2]:
        CE[10,4:9] = 37
    print('CE:\n',CE.astype(int),'\n')

    print('==== DEVELOP ===')
    T_MAX = T_MAX_LIST[example_n] 

    P = 1
    if example_n in [1,2]:
        split_mode = None
        if example_n==2: split_mode = 'split2'
        CE, age = develop(T_MAX, Nj, nBSG, L, J, GE, CE, age, P, verbose=verbose,
            growth_mode='growth3', split_mode=split_mode)


        if example_n == 2:
            print('\nAPPLY SURGERY HERE')
            CE[2:7,4:9] = 1
            CE[7,4:9] = 21
            # print(CE.astype(int))
            print_in_alphabet(CE, nBSG, ALPH=None)
            CE, age = develop(2, Nj, nBSG, L, J, GE, CE, age, P, verbose=verbose,
                growth_mode='growth3', split_mode=split_mode)

            print('\nAPPLY MORE SURGERY HERE')
            CE[4:6,4:9] = 1
            CE[9:12,4:9] = 1
            CE[8,4:9] = 29
            # print(CE.astype(int))
            print_in_alphabet(CE, nBSG, ALPH=None)
            CE, age = develop(2, Nj, nBSG, L, J, GE, CE, age, P, verbose=verbose,
                growth_mode='growth3', split_mode=split_mode)

            print('\nLAST PART')
            CE = np.ones(shape=(20,20))
            CE[4,4:10] = [13,13,14,13,13,13]
            CE[5,4:10] = 19
            CE[6,4:10] = 21
            CE[7,4:10] = 27
            CE[8,4:10] = 29
            CE[6,8] = 29
            CE[8,5] = 13
            print_in_alphabet(CE, nBSG, ALPH=None)
            print(CE.astype(int))
            CE, age = develop(1, Nj, nBSG, L, J, GE, CE, age, P, verbose=verbose,
                growth_mode='growth3', split_mode=split_mode)            
            