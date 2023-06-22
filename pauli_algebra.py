import numpy as np


def pauli_product(pauliL,pauliR):
    prod = np.zeros(len(pauliL),dtype=int)
    coeff = 1
    for i in range(len(pauliL)):
        if pauliL[i] == 0:
            prod[i] = pauliR[i]
        elif pauliR[i] == 0:
            prod[i] = pauliL[i]
        elif pauliL[i] != pauliR[i]:
            prod[i] = list( set([1,2,3]).difference([pauliL[i],pauliR[i]]) )[0]
            if (pauliR[i] - pauliL[i])%3 == 1:
                coeff = coeff*1j
            elif (pauliR[i] - pauliL[i])%3 == 2:
                coeff = coeff*(-1j)
    return prod, coeff



def majorana(whichMajorana,N,encoding):
    # e.g. whichMajorana = [5,3,2,0] means \chi_5 \chi_3 \chi_2 \chi_0
    # whichPauli = [3,3,0,1] means Z_3 Z_2 I_1 X_0
    # N is the number of fermions.
    # note: convention is that least significant bit is 0.
    
    if not hasattr(whichMajorana, '__iter__'):
        whichMajorana = [whichMajorana]
    
    if encoding == 'jordan_wigner':
        pauli_op, coef =  jordan_wigner(whichMajorana,N)
    elif encoding == 'bravyi_kitaev':
        pauli_op, coef =  bravyi_kitaev(whichMajorana,N)
    if len(pauli_op.shape) > 1:
        pauli_op = pauli_op[0]
    return pauli_op, coef

def jordan_wigner(whichMajorana,N):
    # e.g. whichMajorana = [5,3,2,0] means \chi_5 \chi_3 \chi_2 \chi_0
    # whichPauli = [3,3,0,1] means Z_3 Z_2 I_1 X_0
    # N is the number of fermions.
    # note: convention is that least significant bit is 0.
    
    whichPaulis = np.zeros((len(whichMajorana),N//2),dtype=int)
    for i in range(len(whichMajorana)):
        type = (whichMajorana[i]%2) + 1
        qubit = whichMajorana[i]//2
        whichPaulis[i,qubit] = type
        for j in range(qubit):
            whichPaulis[i,j] = 3
            
    if len(whichMajorana) == 1:
            whichPauli = whichPaulis
            coeff = 1
    else:
        whichPauli, coeff = pauli_product(whichPaulis[1,:],whichPaulis[0,:])
        for i in range(2,len(whichMajorana)):
            whichPauli,new_coeff = pauli_product(whichPaulis[i,:],whichPauli);
            coeff = coeff*new_coeff
                
    return whichPauli, coeff


def bravyi_kitaev(whichMajorana,N):
    # e.g. whichMajorana = [5,3,2,0] means \chi_5 \chi_3 \chi_2 \chi_0
    # whichPauli = [3,3,0,1] means Z_3 Z_2 I_1 X_0
    # N is the number of fermions.
    # note: convention is that least significant bit is 0.
    
    def ones_str(num_ones):
        if num_ones >= 1:
            str = '1'
            for _ in range(num_ones-1):
                str += '1'
        else:
            str = ''
        return str

    def partial_order(i,n):
        # returns all j >= i, using the partial order above Eq. 19 in the
        # Bravyi-Kitaev paper. n is the number of qubits
        j = set()
        i_binary = np.binary_repr(i,n)
        i_binary = i_binary[::-1] # flip so that the zeroth element is least significant.
        for l0 in range(n):
            j_l0 = ones_str(l0) + i_binary[l0:]
            j_l0 = int(j_l0[::-1],2) # flip back and convert back to int
            if j_l0 < n:
                j.add(j_l0)
        return j
        
        
    def L_set(i,n):
        # returns the elements in the set L from Eq. 21 of the Bravyi-Kitaev paper
        i_binary = np.binary_repr(i,n)
        i_binary = i_binary[::-1]
        k = set()
        for l0 in range(n):
            if i_binary[l0] == '0':
                continue
            elif i_binary[l0] == '1':
                k_l0 = ones_str(l0) + '0' + i_binary[l0+1:]
                k_l0 = int( k_l0[::-1],2 )
                if k_l0 < n:
                    k.add(k_l0)
        return k
        
    
    whichPaulis = np.zeros((len(whichMajorana),N//2),dtype=int)
    for i in range(len(whichMajorana)):
        type = whichMajorana[i]%2
        qubit = whichMajorana[i]//2
        x_indices = partial_order(qubit,N//2)
        if type == 0:
            z_indices = L_set(qubit,N//2)
        elif type == 1:
            z_indices = L_set(qubit+1,N//2)
        y_indices = z_indices.intersection(x_indices)
        
        for x in x_indices:
            whichPaulis[i,x] = 1
        for z in z_indices:
            whichPaulis[i,z] = 3
        for y in y_indices:
            whichPaulis[i,y] = 2
            
        if len(whichMajorana) == 1:
            whichPauli = whichPaulis
            coeff = 1
        else:
            whichPauli, coeff = pauli_product(whichPaulis[1,:],whichPaulis[0,:])
            for i in range(2,len(whichMajorana)):
                whichPauli,new_coeff = pauli_product(whichPaulis[i,:],whichPauli);
                coeff = coeff*new_coeff
                
    return whichPauli, coeff
