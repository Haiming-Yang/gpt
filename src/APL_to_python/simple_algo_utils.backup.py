import numpy as np

"""
Python translations of the code in chapter 4 of the textbook
Grenander, U. General Pattern Theory: A Mathematical Study of Regular Structures. Oxford Mathematical Monographs. Clarendon Press, 1993. ISBN 9780198536710.
"""

def code_generator(i,k,nBSG):
    # i is the row index of G0, so that i=alpha+1. 
    # k =1,...,Nj where Nj is the number of neighbors (specified in the topology, J)
    # In the first examples from the textbook, bond values directly point to the generator i.
    n = k + nBSG*(i-1)
    return n

def decode_generator(n, nBSG):
    i = 1+ np.floor((n-1)/nBSG)
    k = n - nBSG * (i-1)
    return np.array([i, k])

def opposite(k, Nj):
    """
    Computes 1-based index of the generator placed spatially opposite to k.
    k is also a 1-based index. 
    """
    z = 1 + (-1+k+Nj/2)%Nj
    return int(z) 

def setBSG(nBSG, Nj):
    BSG = np.zeros((nBSG,Nj)) + np.array(range(1,1+Nj))
    for i in range(1,nBSG):
        BSG[i,:] = np.roll(BSG[i,:],i) # set permutation vector for each group element (each row)
    BSG = BSG.astype(int) # this guy is a matrix containing elements of permutation group (1-index based)
    return BSG

def setEXTG0(nBSG, G0, Nj, BSG):
    nG0 = G0.shape[0] - 1

    nGE = nBSG * (nG0+1)
    GE = np.zeros(shape=(nGE,Nj))
    for i in range(1+nG0):
        for k in range(nBSG):
            GE[k+(i*nBSG),:] = G0[i,list(BSG[k]-1)] # -1 is to adjust for 0-based indexing
    return GE

def compenv(Nj, L, J, GE, CE, MODE=None):
    ENV = np.zeros(shape=(Nj,L,L))
    for k in range(Nj):
        if MODE is None:
            temp = np.roll(CE,-J[k,1],axis=1) # roll left by J[k,1]
            temp = np.roll(temp,-J[k,0],axis=0) # roll up by J[k,0]

            # column-based indexing using temp
            ENV[k] = GE[temp.astype(int)-1,opposite(k+1,Nj)-1] # both +1 and -1 for 0-index adjustment
        elif MODE == 'revised':
            roll_right = J[k,0]
            roll_down =  -J[k,1]
            temp = np.roll(CE,roll_right,axis=1) # roll right by roll_right
            temp = np.roll(temp,roll_down,axis=0) # roll down by roll_down

            # column-based indexing using temp
            ENV[k] = GE[temp.astype(int)-1,k] 
    return ENV

def develop(T_MAX, 
    Nj, nBSG, L, J, GE, CE, age,
    P, COMPENV_MODE=None ,verbose=0,ALPH=None):
    # in the textbook T_MAX is named MORE
    if ALPH is None:
        ALPH=' ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    for T1 in range(T_MAX):
        ENV = compenv(Nj, L, J, GE, CE, MODE=COMPENV_MODE)
        if T1%P==0:
            print('ITERATION:%s'%(str(T1+1)))
            if verbose>=50:
                print('T:%s\nCE (before growth):\n%s'%(str(T1),str(CE)))
            if verbose>=100:
                print('ENV:\n%s'%(str(ENV)))

        CE, age = growth1(CE, age, ENV, nBSG,Nj, verbose=verbose)
        
        age = age + 1

        if T1%P==0:
            print('After growth:')
            print_in_alphabet(CE, nBSG, ALPH)
    
    if verbose>=100: print('T %s\n'%(str('final')),CE)

def print_in_alphabet(CE, nBSG, ALPH):
    Z, V1, V2 = compnonzero(CE>nBSG)
    CEn = CE[V1[0]-1:V1[-1],V2[0]-1:V2[-1]]
    decoded = decode_generator(CEn,nBSG)[0,:,:]
    a = replace_index_array_with_values_from(ALPH,decoded.astype(int) -1) # -1 for for 0-based indexing
    
    chararray_letterwise_print(a)    

def growth1(A, age, ENV, nBSG,Nj, verbose=0):
    # see p. 204 of textbook for the growth specification
    # (1) change only at unoccupied sites 
    # (2) new cell at unoccupied sites only if exactly 1 competes for it
    CHANGE = np.logical_and((A<=nBSG),1==np.sum(ENV>0,axis=0))
    
    D = np.sum(ENV, axis=0) # collapse the axis; because of condition (2), no worry getting insensible sum of elements
    if verbose>=200:
        print('D, stores bond values, including the ones not in CHANGED (later discarded)\n', D)

    tempr = np.array(range(1,1+Nj))
    E = np.tensordot(tempr,(ENV>0).astype(int),axes=1)

    B = code_generator(D,E,nBSG) # D from G0, E from neighbor. We also call D the generators and E the NEIGH
    if verbose>=200:
        print('E, stores neighbor indices, including the ones not in CHANGED (later discarded)\n', E) 
        print('CHANGE\n', CHANGE.astype(int))
        print('D * CHANGE\n', D*CHANGE)
        print('E * CHANGE\n',E*CHANGE)
        print('B * CHANGE :\n',B* CHANGE)
    B = (A*(1-CHANGE)) + (CHANGE*B)
    age = age*(1-CHANGE)
    return B, age


def compnonzero(M):
    # display utility
    M1 = M
    M = (M!=0).astype(int)

    temp = []
    N = M.shape[0]
    for i in range(N):
        temp.append(np.any(M[i]).astype(int))
    temp = np.array(temp)
    P = np.where(temp==1)[0][0] + 1 # smallest index that is True (i.e. not zero). +1 because is 1-based index
    Q = np.where(temp[::-1]==1)[0][0] + 1
    V1 = -1+P + range(1,1+2+N-(P+Q)) 

    temp = []
    N = M.shape[1]
    for i in range(N):
        temp.append(np.any(M[:,i]).astype(int))
    temp = np.array(temp)
    P = np.where(temp==1)[0][0] + 1 # smallest index that is True. +1 because is 1-based index
    Q = np.where(temp[::-1]==1)[0][0] + 1
    V2 = -1+P + range(1,1+2+N-(P+Q))
    # print(V2)

    Z = M1[V1[0]-1:V1[-1],V2[0]-1:V2[-1]]
    return Z, V1, V2

def replace_index_array_with_values_from(x,index_array):
    index_array = index_array.astype(int)
    all_elems = set(index_array.reshape(-1))
    y = np.chararray(shape=index_array.shape, unicode=1)
    for el in all_elems:
        y[index_array==el] = r'%s'%(str(x[el]))
    return y

def chararray_letterwise_print(x):
	# assume x is 2D char array
	for y in x:
		this_str = ''
		for z in y:
			if z=='':
				this_str+=' '
			else:
				this_str += z 
		print(this_str)
