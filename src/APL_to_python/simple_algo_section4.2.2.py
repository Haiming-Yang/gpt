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
    AVAILABLE_EXAMPLES = [1,2,3,4,5,6,7,8,9,10]
    SECTION_LABELS = [None,1,2,2,3,3,4,5,6,7,8]
    PAGES = [None, 205,207,209,210,211,213,214,215,217,219]
    T_MAX_LIST = [None,4,6,7,16,4,8,4,9,6,4]

    try:
        print('Example from Ulf Grenander\' textbook section 4.2.2.%s page %s'%(
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
    if example_n in [4]:
        L = 32
    elif example_n in [8,9]:
        L = 15
    else:
        L = 10 # 10 by 10 lattice
    nBSG = 4
    BSG = setBSG(nBSG, Nj)
    print('BSG:\n',BSG,'\n')



    print("==== SETG0 ====")
    age = np.zeros(shape=(L,L))
    if example_n in [1]:
        G0 = np.array([[0,0,0,0],[2,0,2,0]])
    elif example_n in [2]:
        G0 = np.array([[0,0,0,0],[3,0,3,0],[4,0,4,0],[0,0,0,0]])
    elif example_n in [3]:
        G0 = np.array([[0,0,0,0],[3,0,3,0],[4,0,4,0],[5,0,5,0],[0,6,0,6],[6,0,6,0]])
    elif example_n in [4]:
        G0 = np.array([[0,0,0,0],[2,2,2,2]])
    elif example_n in [5]:
        G0 = np.array([[0,0,0,0],[3,3,3,3],[0,0,3,3]])
    elif example_n in [6,7]:
        generator_list = [[0,0,0,0],[3,0,4,0],[0,0,0,5],[0,6,0,0],[0,3,0,0],[0,0,0,4]]
        if example_n == 7:
            generator_list[1][3] = 7
            generator_list.append([7,0,7,0])
        G0 = np.array(generator_list)
    elif example_n in [8]:
        G0 = np.array([[0,0,0,0],[3,3,3,3],[3,0,4,0],[4,0,5,0],[0,3,0,0]])
    elif example_n in [9]:
        G0 = np.array([[0,0,0,0],[3,0,3,0],[4,0,4,0],[3,5,3,5],[5,0,5,0]])
    elif example_n in [10]:
        G0 = np.array([[0,0,0,0],[3,3,3,3],[0,0,3,3],[0,0,0,0]])

    nG0 = G0.shape[0]-1 # no of generators, excluding emtpy generator
    alpha = range(nG0+1) 
    print('G0:\n',G0,'\n')
    print('alpha:\n',alpha,'\n')



    print("==== SETEXTG0 =====")
    GE = setEXTG0(nBSG, G0, Nj, BSG)
    print('GE:\n',GE,'\n')



    print("==== SETCINIT ====")
    CE = np.ones(shape=(L,L))
    if example_n in [1,2,3,5,6,7]:
        CE[4,4] = 5
    else:
        CE[int(L/2),int(L/2)] = 5
    print('CE:\n',CE,'\n')



    print('==== DEVELOP ===')
    T_MAX = T_MAX_LIST[example_n] 

    P = 1
    if example_n in range(10):
        develop(T_MAX, Nj, nBSG, L, J, GE, CE, age, P, verbose=verbose)
    elif example_n in [10]:
        age1 = 1
        develop(T_MAX, Nj, nBSG, L, J, GE, CE, age, P, verbose=verbose, 
            growth_mode='growth2', age1=age1)

