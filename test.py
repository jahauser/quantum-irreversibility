import numpy as np
import circuit, tools, tensor_gate, matrix_gate
import time

gate_constructor = matrix_gate.MatrixGate

def collapse(circuit):
    prod = circuit[0]
    for gate in circuit[1:]:
        prod = prod * gate

    return prod

if __name__ == "__main__":
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    t5 = []
    for counter in range(100):
        print(counter)
        c1 = circuit.Circuit.rand_circuit(gate_constructor, tools.clifford_set, 8, 64)
        c2 = circuit.Circuit.rand_circuit(gate_constructor, tools.clifford_set, 8, 64)
        c3 = circuit.Circuit(c1 + c2)

        t1.append(time.time())
        p1 = collapse(c1)
        t2.append(time.time())
        p2 = collapse(c2)
        t3.append(time.time())
        p3a = p1 * p2
        t4.append(time.time())
        p3b = collapse(c3)
        t5.append(time.time())

    d1 = np.mean([t2[i]-t1[i] for i in range(len(t1))])
    d2 = np.mean([t3[i]-t2[i] for i in range(len(t1))])
    d3 = np.mean([t4[i]-t3[i] for i in range(len(t1))])
    d4 = np.mean([t5[i]-t4[i] for i in range(len(t1))])
    print(f'p1: {d1}\np2: {d2}\np3: {d3}\np4: {d4}')
