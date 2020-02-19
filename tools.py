import numpy as np
import time

zero = np.array([1.0, 0.0])
one  = np.array([0.0, 1.0])

P0 = np.outer(zero, zero)
P1 = np.outer(one, one)

pauli_X = np.array([[0, 1],
                    [1, 0]])

pauli_Y = np.array([[0,-1j],
                    [1j,0]])

phase_gate = lambda phi : np.array([[1, 0],
                                    [0, np.exp(1j*phi)]])

gate_matrices= {"H": 1.0 / 2 ** 0.5 * np.array([[1, 1],
                                                [1, -1]]),
                "T": phase_gate(np.pi / 4),
                "S": phase_gate(np.pi / 2),
                "CNOT": np.tensordot(P0, np.eye(2), axes=0) +
                        np.tensordot(P1, pauli_X, axes=0),
                "P0": P0,
                "P1": P1,
                "X": pauli_X,
                "Y": pauli_Y,
                "NOT": pauli_X}

clifford_set = {k:v for k,v in gate_matrices.items() if k in ["H", "S", "CNOT"]}
universal_set = {k:v for k,v in gate_matrices.items() if k in ["H", "T", "CNOT"]}

def multi_kron(head, *rest):
    if not rest:
        return head
    return np.kron(head, multi_kron(rest))
