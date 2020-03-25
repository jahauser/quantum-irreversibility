import numpy as np
from numba import jit, njit

import tools

@njit()
def main(A,B):
    return np.matmul(A,B)


if __name__ == '__main__':
    main([1],[2])


'''
def compose(self, other):
    sums1 = []
    sums2 = []
    sites1 = self.sites.copy()
    sites2 = other.sites.copy()

    index = {}

    for i, site in enumerate(self.sites):
        if site in other.sites:
        sums1.append(2*i+1)
        sums2.append(2*other.sites.index(site))

        index[2*i+1] = 2*other.sites.index(site)+1

        sites2.remove(site)
    td = np.tensordot(self.representation, other.representation, axes=(sums1, sums2))
    new_sites = sites1 + sites2

    base = list(range(len(td.shape)))
    base.reverse()
    transposition = []
    for i in sums1:
        base.remove(index[i]+2*self.dim-len(sums1)-len(sums2))
    for i in range(len(td.shape)):
        if i in sums1:
            coord = index[i]+2*self.dim-len(sums1)-len(sums2)
            transposition.append(coord)
        else:
            transposition.append(base.pop())
    td = np.transpose(td,transposition)
    return TensorGate(td, new_sites, self.N)
'''
