import numpy as np
from numba import jit, njit, jitclass
import gate

spec = [
    ('representation', 

@jitclass(spec)
class TensorGate(gate.Gate):
    '''Implements Gate using numpy tensors under the hood.'''

    def __init__(self, gate_matrix, sites, N, name=None):
        if any([x >= N for x in sites]):
            raise ValueError('Acts on site outside qubit range')
        self.dim = int(len(gate_matrix.shape)/2)
        gate.Gate.__init__(self, gate_matrix, sites, N, name)
	
    @njit()
    def compose_with(self, other):
        '''Uses np.tensordot to multiply with another tensor.

        This multiplication process involves summing over the correct indices to match how matrix
        multiplication should work. Furthermore, the process requires reordering the tensor
        components afterwards so the tensor continues to act on the proper sites.'''

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

    @njit()
    def dagger(self):
        '''Returns the Hermitian conjugate of a gate and changes its name appropriately.'''

        if self.name:
            if self.name[-1] == '†':
                new_name = self.name[:-1]
            else:
                new_name = self.name + '†'
        transposition = []
        for i in range(self.dim):
            transposition.extend([2*i+1, 2*i])
        return TensorGate(np.transpose(self.representation.conjugate(), transposition), self.sites,
                self.N, name=new_name)
    
    @njit()
    def trace(self):
        '''Uses np.einsum to take the trace of the gate's tensor.'''

        return np.einsum(self.representation,
                [n for n in range(len(self.representation.shape) // 2) for i in range(2)])
