import numpy as np
import gate
import tools

class MatrixGate(gate.Gate):
    '''Implements Gate using matrix tensors under the hood.'''

    def __init__(self, gate_matrix, sites, N, name=None, representation=None):
        if any([x >= N for x in sites]):
            raise ValueError('Acts on site outside qubit range')
        if representation is None:
            if len(sites) == 1:
                before = np.eye(2**sites[0])
                after = np.eye(2**(N-sites[0]-1))
                representation = tools.multi_kron(before, gate_matrix, after)
            elif name == 'CNOT':
                if sites[0] < sites[1]:
                    before = np.eye(2**sites[0])
                    after = np.eye(2**(N-sites[1]-1))
                    mid_d = sites[1]-sites[0]
                    middle = np.kron(tools.P0, np.eye(2**mid_d)) + \
                        tools.multi_kron(tools.P1, np.eye(2**(mid_d-1)), tools.gate_matrices["NOT"])
                else:
                    before = np.eye(2**sites[1])
                    after = np.eye(2**(N-sites[0]-1))
                    mid_d = sites[0]-sites[1]
                    middle = np.kron(np.eye(2**mid_d), tools.P0) + \
                        tools.multi_kron(tools.gate_matrices["NOT"], np.eye(2**(mid_d-1)), tools.P1)
                representation = tools.multi_kron(before, middle, after)
            else:
                raise ValueError("MatrixGate can't build a gate out of that")

        gate.Gate.__init__(self, representation, sites, N, name)

    def compose_with(self, other):
        sites1 = self.sites.copy()
        sites2 = other.sites.copy()
        new_sites = sites1 + sites2

        prod = np.matmul(self.representation, other.representation)

        return MatrixGate(None, new_sites, self.N, representation=prod)

    def dagger(self):
        '''Returns the Hermitian conjugate of a gate and changes its name appropriately.'''
        new_name = None
        if self.name:
            if self.name[-1] == '†':
                new_name = self.name[:-1]
            else:
                new_name = self.name + '†'
        return MatrixGate(None, self.sites, self.N, name=new_name,
                representation=self.representation.conjugate().T)

    def trace(self):
        return np.trace(self.representation)

    def trace2(self, other):
        return np.trace(self.compose_with(other))
