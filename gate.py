class Gate:
    '''An abstract gate class, designed to be inherited by particular implementations.

    Right now, everything is built for TensorGate. However, in the future I'd like to use
    MatrixGate (which will use normal matrices as representations) to check correctness, and 
    SparseMatrixGate (which will use sparse matrices as representations) to see if this works
    faster than TensorGate.'''

    def __init__(self, representation, sites, N, name=None):
        self.representation = representation
        self.sites = sites
        self.N = N
        self.name = name

    def __mul__(self, other):
        '''Multiplies by another gate using (*).'''

        return self.compose_with(other)

    def compose_with(self, other):
        '''Composes with another gate.'''

        pass

    def apply_to(self, state):
        '''Applies this gate to a quantum state.'''

        pass

    def dagger(self):
        '''Returns the Hermitian conjugate of this gate.'''

        pass

    def trace(self):
        '''Returns the trace of this gate.'''

        pass

    def __str__(self):
        if self.name:
            return self.name + str(self.sites)
        return self.representation.__str__()

    def __repr__(self):
        if self.name:
            return self.name + str(self.sites)
        return self.representation.__repr__()
