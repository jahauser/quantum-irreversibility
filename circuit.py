import numpy as np
import re

class Circuit(list):
    '''List wrapper for quantum gates.

    Primary feature currently is dagger() method which provides an efficient way for storing
    the Hermitian conjugate of circuits.'''

    def __init__(self, *args):
        list.__init__(self, *args)

    def dagger(self):
        '''Uses property that (UV)^\dag = V^\dag U^\dag.'''

        return Circuit(gate.dagger() for gate in self[::-1])

    def __getitem__(self, *args):
        '''Accesses gates in circuit using circuit[n] syntax.

        Both cases use list's __getitem__, but the latter case ensures that a Circuit is returned
        when slices are used for __getitem__.'''

        if isinstance(args[0], int):
            return list.__getitem__(self, *args)
        return Circuit(list.__getitem__(self, *args))

    def product(self):
        prod = self[0]
        for gate in self[1:]:
            prod = prod * gate
        return prod
    
    @staticmethod
    def from_string(gate_constructor, gate_map, N, circuit_string):
        p = re.compile("([A-Z]*\[\d(?:, \d)*\])")
        gate_strings = p.findall(circuit_string)

        gates = []

        for gate in gate_strings:
            name = gate[:gate.find('[')]
            sites = gate[gate.find('['):]
            matrix = gate_map[name]
            gates.append(gate_constructor(matrix, [int(site) for site in
                sites.strip('][').split(', ')], N, name=name))
        return Circuit(gates)

    @staticmethod
    def rand_circuit(gate_constructor, gate_map, N, num_gates):
        '''Generates a circuit composed of random gates.

        num_gates gates are generated of type gate_constructor, selected from those in a dictionary
        (with keys corresponding to gate names and values corresponding to matrices). The sites
        they act on are randomly selected from those possible for a N-qubit system.'''

        gates = []
        for i in range(num_gates):
            name = np.random.choice(list(gate_map.keys()))
            matrix = gate_map[name]
            dim = int(len(matrix.shape) / 2)
            sites = list(np.random.choice(np.arange(N), size=dim, replace=False))
            gates.append(gate_constructor(matrix, sites, N, name=name))
        return Circuit(gates)
