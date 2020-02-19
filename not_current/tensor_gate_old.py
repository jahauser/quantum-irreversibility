# -*- coding: utf-8 -*-

import numpy as np
import tools
import sys
import time

phase_gate = lambda phi : np.array([[1, 0],
                                    [0, np.exp(1j*phi)]])

gate_matrices= {"H": 1.0 / 2 ** 0.5 * np.array([[1, 1],
                                                [1, -1]]),
                "T": tools.phase_gate(np.pi / 4),
                "S": tools.phase_gate(np.pi / 2),
                "CNOT": np.tensordot(tools.P0, np.eye(2), axes=0) +
                        np.tensordot(tools.P1, tools.pauli_X, axes=0),
                "P0": tools.P0,
                "P1": tools.P1,
                "X": tools.pauli_X,
                "Y": tools.pauli_Y,
                "NOT": np.array([[0, 1],
                                 [1, 0]])}



def rand_gates(gate_set, N, num_gates):
    k = 0
    while k < num_gates:
        gate = np.random.choice(gate_set)
        matrix = gate_matrices[gate]
        dim = int(len(matrix.shape) / 2)
        yield TensorGate(matrix, list(np.random.choice(np.arange(N), size=dim, replace=False)), dim=dim, name=gate)
        k += 1


def gate(name, bits):
    return TensorGate(gate_matrices[name], bits, name=name)



class TensorGate:
    def __init__(self, matrix, bits, dim=None, name=None):
        self.matrix = matrix
        self.bits = bits
        if not dim:
            self.dim = int(len(self.matrix.shape)/2)
        else:
            self.dim = dim

        self.name = name

    def __mul__(self, other):
        sums1 = []
        sums2 = []
        bits1 = self.bits.copy()
        bits2 = other.bits.copy()

        index = {}

        for i, bit in enumerate(self.bits):
            if bit in other.bits:
                sums1.append(2*i+1)
                sums2.append(2*other.bits.index(bit))

                index[2*i+1] = 2*other.bits.index(bit)+1

                #bits1.remove(bit)
                bits2.remove(bit)
        td = np.tensordot(self.matrix, other.matrix, axes=(sums1, sums2))
        new_bits = bits1 + bits2

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
        return TensorGate(td, new_bits)

    #@profile
    def apply_to(self, state):
        my_comps = [2*k+1 for k in range(self.dim)]

        N = len(state.shape)
        transposition = []
        index = {bit: num for (num, bit) in enumerate(self.bits)}
        base = list(range(len(self.bits),N))
        for i in range(N):
            contained = i in self.bits
            if contained:
                transposition.append(index[i])
            else:
                transposition.append(base.pop(0))

        td = np.tensordot(self.matrix, state, axes=(my_comps, self.bits))
        return np.transpose(td,transposition)

    def dagger(self):
        if self.name[-1] == '†':
            new_name = self.name[:-1]
        else:
            new_name = self.name + '†'
        transposition = []
        for i in range(self.dim):
            transposition.extend([2*i+1, 2*i])
        return TensorGate(np.transpose(self.matrix.conjugate(), transposition), self.bits, dim=self.dim, name=new_name)

    def __str__(self):
        if self.name:
            return self.name + str(self.bits)
        return self.matrix.__str__()

    def __repr__(self):
        if self.name:
            return self.name + str(self.bits)
        return self.matrix.__repr__()


class GateSequence(list):
    def __init__(self, *args):
        list.__init__(self, *args)

    def apply_to(self, state):
        for gate in self:
            state = gate.apply_to(state)
        return state

    def apply_to2(self, state):
        if not self:
            return state
        return self[1:].apply_to(self[0].apply_to(state))

    def dagger(self):
        return GateSequence(gate.dagger() for gate in self[::-1])

    def __getitem__(self, *args):
        if isinstance(args[0], int):
            return list.__getitem__(self, *args)
        return GateSequence(list.__getitem__(self, *args))

def product(gates):
    if type(gates[0]) == list:
        return product([product(sub_gates) for sub_gates in gates])
    prod = gates[0]
    for sub_gate in gates[1:]:
        prod = prod * sub_gate
    return prod

def collapse_aux(gates, i):
    M = len(gates)
    #for j in range(i+1, M):


def new_rand_gates(gate_set, N, num_gates):
    bit_list = []
    #for n in range(num_gates):

    k = 0
    while k < num_gates:
        gate = np.random.choice(gate_set)
        matrix = gate_matrices[gate]
        dim = int(len(matrix.shape) / 2)
        yield TensorGate(matrix, list(np.random.choice(np.arange(N), size=dim, replace=False)), dim=dim, name=gate)
        k += 1

# pick gate

'''
gates1 = [gate("CNOT",[i,i+1]) for i in range(12)]+ [gate("CNOT",[i,i+1]) for i in range(12)][:-1]
gates2 = [[gate("CNOT",[i,i+1]) for i in range(12)],[gate("CNOT",[i,i+1]) for i in range(12)]]

N = 13
p = product([gate("CNOT",[i,i+1]) for i in range(N)])
t1 = time.time()
p*gate("H",[N,N])
t2 = time.time()

print(t2-t1)

t1 = time.time()
for i in range(10*400):
    gate = TensorGate(np.random.rand(16).reshape(2,2,2,2), [1,2])
    for j in range(2*10*400):
        new_gate = TensorGate(np.random.rand(16).reshape(2,2,2,2), [1,2])
        gate = gate * new_gate
t2 = time.time()

print(t2-t1)
'''
