import numpy as np
import tools
import sys
from tensor_gate import *

t = 3

#K = 200
N = 8
#gate_scale = 10

clifford_set = ["CNOT", "H", "S"]
universal_set = ["CNOT", "H", "T"]

architecture = "all clifford"


for gate_scale in [1]:
    for K in [100, 100, 100, 100, 100]:
        circuits = []
        for k in range(K):

            #if k % 2 == 0:
            #    print(k)

            num_gates = int(gate_scale*(N**2))

            if architecture == "all clifford":
                U = GateSequence(rand_gates(clifford_set, N, num_gates)) # how many should we really apply?
            elif architecture == "all universal":
                U = GateSequence(rand_gates(universal_set, N, num_gates)) # how many should we really apply?
            elif architecture == "new":
                V1 = GateSequence(rand_gates(clifford_set, N, num_gates))
                V2 = GateSequence(rand_gates(["T"], N, 1))
                U = GateSequence(V1 + V2 + V1.dagger() + V2.dagger())
            elif architecture == "doped":
                U1 = GateSequence(rand_gates(clifford_set, N, num_gates))
                U2 = GateSequence(rand_gates(clifford_set, N, num_gates))
                V = GateSequence(rand_gates(["T"], N, 1))
                U = GateSequence(U1+V+U2)

            circuits.append(U)

        #print(circuits)

        Phi = 0.0

        for i in range(K):
            #if i % 2 == 0:
            #    print(i)

            for j in range(K):
                circuit = circuits[i]+circuits[j].dagger()
                prod = circuit[0]
                for gate in circuit[1:]:
                    prod = gate * prod
                #print(prod.matrix.shape)
                trace = np.einsum(prod.matrix, [n for n in range(len(prod.matrix.shape) // 2) for i in range(2)])
                Phi += np.abs(trace)**(2*t)

        Phi = Phi / (K**2)

        print(Phi)

        with open('test.txt', 'a') as output_file:
            output_file.write(f't: {t}, gate_scale: {gate_scale}, K: {K}, N: {N}, Architecture: {architecture}, Phi: {Phi}\n')


# run as is new circuit
# store separate matrices for each index and only combine as needed
# separate and use commutation relations
