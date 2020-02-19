import numpy as np
import gate
import tools

class MatrixGate(gate.Gate):
    def __init__(self, gate_matrix, sites, N, name=None):
        if any([x >= N for x in sites]):
            raise ValueError('Acts on site outside qubit range')

        if len(sites) == 1:
            representation = tools.multi_kron(np.eye(sites[0]), gate_matrix, np.eye(N-1-sites[0]))
        elif len(sites) == 2:
            if name == "CNOT":
                middle = np.kron(tools.P0, eye(2**(N-1))) + 
                            tools.multi_kron(tools.P1, eye(2**(N-2)), tools.pauli_X)
                representation = tools.multi_kron(np.eye(sites[0]), middle, np.eye(N-1-sites[1]))
            else:
                raise ValueError('Unknown 2-qubit operator')
        else:
            raise ValueError('Unknown >2-qubit operator')

        gate.Gate.__init__(representation, sites, N, name)

    def compose_with(self, other):
        return np.matmul(self.representation, other.representation)

    def apply_to(self, other):
        return np.matmul(self.representation, other.representation)

    def dagger(self):
        self.representation = self.representation.conj().T
        return self
