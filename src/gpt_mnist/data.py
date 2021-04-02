import numpy as np

"""
The vocabulary used here follows as much as possible chapter 4.2 of the textbook
"Ulf Grenander. General Pattern Theory: A Mathematical Study of Regular Structures. Oxford Mathematical Monographs. Clarendon Press, 1993. 
ISBN 9780198536710. URL https://books.google.com.sg/books?id=Z-8YAQAAIAAJ"
"""

def code_generator(i, k_neighbor,nBSG):
    # technically i is a bond value. Bond value is used in the first textbook example as alpha+1
    # k_neighbor =1,...,Nj where Nj is the number of neighbors (specified in the topology, J)
    n = nBSG*(i-1) + k_neighbor
    return n

def decode_generator(n, nBSG):
    i = 1+ np.floor((n-1)/nBSG)
    k = n - nBSG * (i-1)
    return np.array([i, k])

def opposite(k, Nj):
    z = 1 + (-1+k+Nj/2)%Nj
    return int(z) 

class SimpleGen():
    def __init__(self, L, G0=None, topology_matrix=None, 
        compenv_mode=None, growth_mode=None, verbose=0):
        super(SimpleGen, self).__init__()

        # don't worry about this
        self.compenv_mode = compenv_mode
        self.growth_mode = growth_mode

        # assume square board of side length L
        self.L = L

        # Assume topology of a 2D board. Hence shape is (N_neighbor, 2)
        self.topology_matrix = topology_matrix
        if topology_matrix is None: 
            self.topology_matrix = np.array([[0,1],[1,0],[0,-1],[-1,0]]) # TOP, RIGHT, BOTTOM, LEFT
        self.N_neighbor = self.topology_matrix.shape[0]

        # Assume BSG contains elements that permute neighbors cyclically. 
        # note: does depend on the topology_matrix.
        self.nBSG = self.N_neighbor
        self.BSG = self.setBSG()

        self.G0 = G0
        if G0 is None:
            self.G0 = np.array([[0,0,0,0],[2,0,2,0]])
        sG0 = self.G0.shape
        self.nG0 = sG0[0]-1
        assert(sG0[1]==self.N_neighbor) # each generator (each row) assigns a bond value for each neighbor 

        self.GE = self.setEXTG0()
        self.nGE = self.GE.shape[0]

        if verbose>=100:
            print('topo matrix or J\n',self.topology_matrix)
            print('BSG\n',self.BSG)
            print('G0\n',self.G0)
            print('GE\n',self.GE)

    def setBSG(self,):
        nBSG, Nj = self.nBSG, self.N_neighbor
        BSG = np.zeros((nBSG,Nj)) + np.array(range(1,1+Nj))
        for i in range(1,nBSG):
            BSG[i,:] = np.roll(BSG[i,:],i) # set permutation vector for each group element (each row)
        BSG = BSG.astype(int) # this guy is a matrix containing elements of permutation group (1-index based)
        return BSG

    def setEXTG0(self):
        nBSG, G0, Nj, BSG = self.nBSG, self.G0, self.N_neighbor, self.BSG
        nG0 = G0.shape[0] - 1

        nGE = nBSG * (nG0+1)
        GE = np.zeros(shape=(nGE,Nj))
        for i in range(1+nG0):
            for k in range(nBSG):
                GE[k+(i*nBSG),:] = G0[i,list(BSG[k]-1)] # -1 is to adjust for 0-based indexing
        return GE

    def develop(self,T_MAX, board, compenv_mode=None, growth_mode='classic'):
        # in the textbook T_MAX is named MORE
        for T1 in range(T_MAX):
            ENV = self.compenv(board, mode=compenv_mode)
            board = self.growth1(board, ENV, mode=growth_mode)
        return board

    def compenv(self, board, mode=None):
        Nj, L, J, GE = self.N_neighbor, self.L, self.topology_matrix, self.GE
        CE = board

        ENV = np.zeros(shape=(Nj,L,L))
        for k in range(Nj):
            if self.compenv_mode is None:
                # use the revised version, so that TOP/BOTTOM/LEFT/RIGHT in the topology 
                #   are exactly TOP/BOTTOM/LEFT/RIGHT in the board.
                roll_right = J[k,0]
                roll_down =  -J[k,1]
                temp = np.roll(CE,roll_right,axis=1) # roll right by roll_right
                temp = np.roll(temp,roll_down,axis=0) # roll down by roll_down

                # column-based indexing using temp
                ENV[k] = GE[temp.astype(int)-1,k] 
            elif self.compenv_mode == 'classic':
                temp = np.roll(CE,-J[k,1],axis=1) # roll left by J[k,1]
                temp = np.roll(temp,-J[k,0],axis=0) # roll up by J[k,0]

                # column-based indexing using temp
                ENV[k] = GE[temp.astype(int)-1,opposite(k+1,Nj)-1] # both +1 and -1 for 0-index adjustment
            else:
                raise RuntimeError('Compenv mode error?')
        return ENV

    def growth1(self, board, ENV, mode='classic'):
        nBSG, Nj = self.nBSG, self.N_neighbor

        if mode is None:
            # assume 8 DIR topology
            n_effective_bonds = np.sum(ENV>0,axis=0)
            CHANGE = np.logical_and((board<=nBSG),1==n_effective_bonds)
            D = np.sum(ENV, axis=0) # collapse the axis; because of condition (2), no worry getting insensible sum of elements
            tempr = np.array(range(1,1+Nj))
            E = np.tensordot(tempr,(ENV>0).astype(int),axes=1)
            B = code_generator(D,E,nBSG) # D from G0, E from neighbor. We also call D the generators and E the NEIGH
            
            TRIHEAD_CHANGE = np.logical_and((board<=nBSG),np.logical_or(n_effective_bonds==2,n_effective_bonds==3,))
            TCHANGE = TRIHEAD_CHANGE.astype(int)*(self.nGE-1)
            # raise Exception('NONON')
            
            SURROUND_CHANGE = np.logical_and((board<=nBSG),n_effective_bonds>=6)
            SCHANGE = SURROUND_CHANGE.astype(int)*(self.nGE-1)

            new_board = (board*(1-CHANGE)*(1- TRIHEAD_CHANGE )*(1- SURROUND_CHANGE)) \
                + (CHANGE*B) + TCHANGE + SCHANGE
        elif mode == 'classic':
            n_effective_bonds = np.sum(ENV>0,axis=0)
            CHANGE = np.logical_and((board<=nBSG),1==n_effective_bonds)
            D = np.sum(ENV, axis=0) # collapse the axis; because of condition (2), no worry getting insensible sum of elements
            tempr = np.array(range(1,1+Nj))
            E = np.tensordot(tempr,(ENV>0).astype(int),axes=1)
            B = code_generator(D,E,nBSG) # D from G0, E from neighbor. We also call D the generators and E the NEIGH
            new_board = (board*(1-CHANGE)) + (CHANGE*B)
        else:
            raise RuntimeError('Invalid growth1() mode.')
        return new_board


########### printing utils ###########

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

class CustomBoardPrinter(object):
    def __init__(self, ALPH=None):
        super(CustomBoardPrinter, self).__init__()
        
        self.ALPH = ALPH
        if ALPH is None: self.ALPH = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    def print_in_alphabet(self, board, nBSG):
        CE = board
        ALPH = self.ALPH

        Z, V1, V2 = self.compnonzero(CE>nBSG)
        CEn = CE[V1[0]-1:V1[-1],V2[0]-1:V2[-1]]
        decoded = decode_generator(CEn,nBSG)[0,:,:]
        a = replace_index_array_with_values_from(ALPH, decoded.astype(int) -1) # -1 for for 0-based indexing
        
        chararray_letterwise_print(a)   

    def compnonzero(self, M):
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


