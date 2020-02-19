import numpy as np
import sys, argparse
import circuit, tools, tensor_gate

gate_constructor = tensor_gate.TensorGate

def get_args():
    '''Retrieve file name arguments.'''
    
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="input file containing test parameters")
    parser.add_argument("outfile", help="output file containing test results")
    parser.add_argument("--new", action='store_true',
            help="flag to indicate whether outfile is new and should have header added")
    args = parser.parse_args()
    return args

def all_clifford_maker(N, circuit_scale):
    '''Return a random circuit composed only of Clifford gates.'''

    num_gates = int(circuit_scale*(N**2))
    return circuit.Circuit.rand_circuit(gate_constructor, tools.clifford_set, N, num_gates)

def all_universal_maker(N, circuit_scale):
    '''Return a random circuit composed of arbitrary unitary gates.'''

    num_gates = int(circuit_scale*(N**2))
    return circuit.Circuit.rand_circuit(gate_constructor, tools.universal_set, N, num_gates)

def frame_potential(t, K, N, circuit_scale, circuit_maker):
    '''Calculate the t^th frame potential for a set of circuits.

    In particular, the set of circuits contains K circuits operating on a space of N qubits.
    A circuit_maker function is given corresponding to the desired circuit architecture, and a
    circuit_scale parameter provided to scale the size of the circuit produced by this function.'''

    # First, we generate a set of circuits
    circuits = []
    for k in range(K):
        circuit = circuit_maker(N, circuit_scale)
        circuits.append(circuit)

    # Next, we initialize our frame potential Phi, which is built up over the following loop
    Phi = 0.0

    # Loop over all pairs of circuits
    for i in range(K):
        for j in range(K):
            # Prepare the circuit U V^\dag
            circuit = circuits[i]+circuits[j].dagger()
            
            # Multiple as if circuit[0] is furthest to the right
            prod = circuit[0]
            for gate in circuit[1:]:
                prod = gate.compose_with(prod)
            
            trace = prod.trace()
            Phi += np.abs(trace)**(2*t)

    Phi = Phi / (K**2)
    return Phi

def parse_input(infile_name):
    '''Parses the input file into the desired test cases.'''

    strings = []
    tests = []
    with open(infile_name, 'r') as infile:
        infile.readline()
        for line in infile:
            line = line.rstrip()
            params = line.split(', ')
            if params[4] == 'all clifford':
                circuit_maker = all_clifford_maker
            elif params[4] == 'all universal':
                circuit_maker = all_universal_maker
            else:
                raise ValueError('Unknown circuit architecture')
            strings.append(line)
            tests.append(tuple(int(param) for param in params[:3]) + \
                    (float(params[3]),) + (circuit_maker,))
    return (strings, tests)

def main(args):
    '''Reads input file and runs tests, outputting after each test.'''

    (strings, tests) = parse_input(args.infile)

    if args.new:
        with open(args.outfile, 'a') as outfile:
            outfile.write('t, K, N, circuit_scale, architecture, Phi\n')

    for (string, test) in zip(strings, tests):
        phi = frame_potential(*test)
        with open(args.outfile, 'a') as outfile:
            outfile.write(string + f', {phi}\n')

if __name__ == '__main__':
    args = get_args()
    main(args)


# notes
# run as is new circuit
# store separate matrices for each index and only combine as needed
# separate and use commutation relations
