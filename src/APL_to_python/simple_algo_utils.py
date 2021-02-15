import numpy as np

"""
Python translations of the code in chapter 4 of the textbook
Grenander, U. General Pattern Theory: A Mathematical Study of Regular Structures. Oxford Mathematical Monographs. Clarendon Press, 1993. ISBN 9780198536710.
"""
np. set_printoptions(threshold=np. inf)
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
    P, COMPENV_MODE=None ,verbose=0,ALPH=None, **kwargs):


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

        if not 'growth_mode' in kwargs:
            CE, age = growth1(CE, age, ENV, nBSG,Nj, verbose=verbose)
        else:
            if kwargs['growth_mode'] == 'growth2':
                CE, age = growth2(CE, age, ENV, nBSG,Nj, age1=kwargs['age1'], verbose=verbose)
            elif kwargs['growth_mode'] == 'growth3':
                split_mode = kwargs['split_mode'] if 'split_mode' in kwargs else None
                CE, age = growth3(CE, age, ENV, nBSG, Nj, J, L, split_mode=split_mode, verbose=verbose)
                
        age = age + 1

        if T1%P==0:
            print('After growth:')
            print_in_alphabet(CE, nBSG, ALPH)

    
    if verbose>=100: print('T %s\n'%(str('final')),CE)
    return CE, age

def print_in_alphabet(CE, nBSG, ALPH=None):
    if ALPH is None:
        ALPH=' ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    Z, V1, V2 = compnonzero(CE>nBSG)
    CEn = CE[V1[0]-1:V1[-1],V2[0]-1:V2[-1]]
    decoded = decode_generator(CEn,nBSG)[0,:,:]
    a = replace_index_array_with_values_from(ALPH,decoded.astype(int) -1) # -1 for for 0-based indexing
    
    chararray_letterwise_print(a)    

def growth1(A, age, ENV, nBSG, Nj, verbose=0):
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

def growth2(A, age, ENV, nBSG, Nj, age1, verbose=0):
    B, age = growth1(A, age, ENV, nBSG,Nj, verbose=verbose)
    CHANGE1 = (B>nBSG).astype(float)*(age>age1).astype(float)
    CHANGE2 = (B>nBSG).astype(float)*(age==age1).astype(float)

    B = (1-CHANGE1-CHANGE2)*B + CHANGE1 + 13* CHANGE2
    return B, age

def growth3(CE, age, ENV, nBSG, Nj, J, L, split_mode=None, verbose=0):
    if split_mode is None:
        NEW = split1(CE, Nj)
    else:
        if split_mode == 'split2': 
            NEW = split2(CE, ENV, Nj, nBSG, L)
            # raise Exception('here!')

    for J1 in range(1,1+Nj):
        while True:
            if np.all(~(NEW[J1-1,:,:]>nBSG)):
                break
            M = list((NEW[J1-1,:,:]>nBSG).reshape(-1).astype(int)).index(1) + 1
            X = int(np.ceil(M/L))
            Y = int(M - L*(X-1))
            # print(J1, M, X, Y)
            # print(NEW.shape)

            J2 = NEW[J1-1,X-1,Y-1]
            NEW[J1-1,X-1,Y-1] = 0
            DIR = J[J1-1,:]

            # print(DIR)
            # print([X, Y]+ list(DIR))
            # raise Exception('WAIT FOR IT!')
            CE, NEW = move1(CE, [X, Y]+ list(DIR), L, nBSG, Nj, NEW)
            CE[X+DIR[0]-1, Y+DIR[1]-1] = J2            
        # print(CE)
    # raise Exception('gg')
    return CE, age



def move1(A, PAR, L, nBSG, Nj, NEW):
    Z = PAR[0:2]
    DIR = PAR[2:4]
    SET = np.tensordot(np.array(range(1,1+L)),DIR,axes=0) \
        + np.tensordot(np.ones(shape=(L,)),Z,axes=0)
    # print(SET)
    ANS = test(SET, L)
    # print(ANS)
    temp = [SET[x-1,:] for x in ANS]
    SET = np.array(temp).astype(int)
    # print(SET)
    temp1 = index_col_and_row_by_list(A, [SET[:,0]-1,SET[:,1]-1])
    V = np.diag(temp1)
    # print(V)
    NONEMPTY = V>nBSG

    def get_last(x):
        # assume x boolean 
        # 1-based index of the first 0 or False found
        i = 0
        for x1 in x:
            i = i+1
            if not x1: return i
        return i+1
    LAST = get_last(NONEMPTY)
    # print('LAST',LAST)

    B = A.copy()
    NEW1 = NEW.copy()
    i=1
    while True:
        B[SET[i,0]-1, SET[i,1]-1] = A[SET[i-1,0]-1,SET[i-1,1]-1]
        NEW1[:,SET[i,0]-1, SET[i,1]-1] = NEW[:,SET[i-1,0]-1,SET[i-1,1]-1]
        NEW1[:,SET[0,0]-1,SET[0,1]-1] = np.zeros(shape=(Nj,))
        i=i+1
        if (i<LAST): continue
        break
    NEW = NEW1
    return B, NEW
    

def test(Z,L):
    ANS = (1<=Z[:,0])*(L>=Z[:,0])*(1<=Z[:,1])*(L>=Z[:,1])
    temp  = np.array(range(1,1+len(ANS)))
    ANS = replicate(ANS, temp) # in APL, replicate is done by symbol /
    return ANS

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


def split1(A, Nj):
    B = 9* np.isin(A,[5,6,7,8]).astype(float)
    B = np.tensordot(np.ones(shape=(Nj)),B, axes=0)
    return B


def split2(CE, ENV, Nj, nBSG, L):
    I = decode_generator(CE, nBSG)[0,:,:]
    K = decode_generator(CE, nBSG)[1,:,:]

    CE, B = tau1(CE, Nj, L, ENV, I)
    NEW = B
    NEW[:,1,:,:] =  1 + np.remainder(NEW[:,1,:,:]+ np.tensordot(np.ones(shape=(nBSG,)), K-2,axes=0), nBSG )
    NEW = code_generator(NEW[:,0,:,:],NEW[:,1,:,:],nBSG)
    return NEW

def tau1(CE, Nj, L, ENV, I, change_mode=None):
    B = np.zeros(shape=(Nj,2,L,L)) 
    for J1 in range(1,1+Nj):
        B[J1-1,0,:,:] = newg2(ENV[J1-1,:,:], I, J1, L)
        B[J1-1,1,:,:] = np.ones(shape=(L,L)) * J1
    if change_mode is None:
        CE = changeI(CE)
    else:
        raise RuntimeError('not implemented')
    return CE, B

def newg2(BETA, I, J1, L):
    Z = np.zeros(shape=(L,L))
    COND = (I>1)*(1<np.abs(BETA-I))
    COND2 = (COND * (I==2)).astype(int)
    COND10 = (COND * (I==10)).astype(int) 
    CONDINT = (COND - (COND2+COND10)).astype(int)
    Z = Z + 3*COND2*(J1==1)*(BETA>3)
    Z = Z + 9*COND10*(J1==3)*(BETA<9)
    Z = Z + CONDINT * np.logical_or(J1==1,J1==3) * (I + np.sign(BETA-I))
    return Z

def changeI(CE):
    # trivial identity transformation. Just a placeholder.
    return CE

######## utils ###########

def replicate(INDEX, A):
    # in APL, this is INDEX / A
    # assume INDEX and A a row vector
    out = []
    for i,a in zip(INDEX,A):
        out = out + [a for j in range(i)]
    return np.array(out)

def index_col_and_row_by_list(x, indices):
    """
    let x be 2D array
    let indices = (ix, iy)  be list of indices.
    then extract all rows r indexed by ix at columns c indexed by iy
    
    x = np.array(range(1,30+1)).reshape(5,6)
    indices = [[1,2,3],[1,3,5]]
    y = index_col_and_row_by_list(x, indices)
    print(x)
    print(y)

    [[ 1  2  3  4  5  6]
     [ 7  8  9 10 11 12]
     [13 14 15 16 17 18]
     [19 20 21 22 23 24]
     [25 26 27 28 29 30]]
         
    [[ 8 10 12]
     [14 16 18]
     [20 22 24]]

    """
    ix, iy = indices
    mx, my = np.meshgrid(ix,iy)
    y = x[mx.reshape(-1),my.reshape(-1)]
    y = y.reshape(mx.shape).T
    return y

