import numpy as np
import circuit, tools, tensor_gate, matrix_gate
gate_string = '[H[0], CNOT[1, 2], CNOT[0, 2], CNOT[0, 1], CNOT[0, 2], CNOT[0, 2], S[2], CNOT[0, 2], H[0], S[2]]'
mat_circuit = circuit.Circuit.from_string(matrix_gate.MatrixGate, tools.gate_matrices, 3, gate_string)
ten_circuit = circuit.Circuit.from_string(tensor_gate.TensorGate, tools.gate_matrices, 3, gate_string)
