"""The custom ansatz module"""
from typing import Optional
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.exceptions import CircuitError

class RVB:
    """A class to help build Resonating Valence Bond ansatz circuit
    """

    def __init__(
            self,
            num_qubits: Optional[int] = None,
            reps: int = 1
    ) -> None:
        self._params = []
        self._num_qubits = num_qubits
        self._reps = reps
        pass

    def new_param(self, paramstr = 'phi') -> Parameter:
        """Create a new parameter and save it."""
        param = Parameter("".join([paramstr, str(len(self._params))]))
        self._params.append(param)
        return param
    
    def add_eSWAP(self):
        circ = QuantumCircuit(2, name="eSWAP")
        circ.cnot(1, 0)
        #circ.crx(phi, control_qubit=0, target_qubit=1)
        circ.crx(self.new_param(), control_qubit=0, target_qubit=1)
        circ.x(0)
        #circ.rz(-phi/2, 0)
        #circ.x(0)
        circ.rz(- self.new_param()/2, 0)
    
        circ.cnot(1,0)
        return circ.to_gate()
    
    def circuit(self) -> QuantumCircuit:
        """Make the quantum circuit."""

        n = self._num_qubits

        if n % 2 != 0:
            raise CircuitError("The number of qubits should be even!")
        

        ansatz = QuantumCircuit(n, name="RVB")

        # build initial state
        for i in range(0,n):
            ansatz.x(i)

        # build bell state
        for j in range(0,n):
            if j % 2 == 0:
                ansatz.h(j)
                ansatz.cx(j,j+1)

        for k in range(self._reps):
            for l in range(n-1,0,-2):
                
                    if l < n-1:
                        ansatz.append(self.add_eSWAP(), [l,l+1])
                    else:
                        ansatz.append(self.add_eSWAP(), [l,0])
                        
            for l in range(0, n-1, 2):
                ansatz.append(self.add_eSWAP(), [l,l+1])

        return ansatz
    

class HVAnsatz:
    """A class to help build Hamiltonian Variational Ansatz circuit
    """

    def __init__(
            self,
            num_qubits: Optional[int] = None,
            reps: int = 1
    ) -> None:
        self._params = []
        self._num_qubits = num_qubits
        self._reps = reps
        pass

    def new_param(self, paramstr = 'phi') -> Parameter:
        """Create a new parameter and save it."""
        param = Parameter("".join([paramstr, str(len(self._params))]))
        self._params.append(param)
        return param
    
    def circuit(self) -> QuantumCircuit:
        """Make the quantum circuit."""

        n = self._num_qubits

        if n % 2 != 0:
            raise CircuitError("The number of qubits should be even!")
        
        # Build a custom ansatz from scratch
        ansatz = QuantumCircuit(n, name="HVA")

        # build initial state
        for i in range(0,n):
            ansatz.x(i)

        # build bell state
        for j in range(0,n):
            if i % 2 == 0:
                ansatz.h(j)
                ansatz.cx(j,j+1)

        # loop for number of layer
        for j in range(self._reps):
            # odd Hamiltonian
            for i in range(n-1,0,-2):
                if i < n-1:
                    ansatz.rzz(self.new_param('phi'),i,i+1)
                    ansatz.ryy(self.new_param('phi'),i,i+1)
                    ansatz.rxx(self.new_param('phi'),i,i+1)
                else:
                    ansatz.rzz(self.new_param('phi'),i,0)
                    ansatz.ryy(self.new_param('phi'),i,0)
                    ansatz.rxx(self.new_param('phi'),i,0)

            # even Hamiltonian
            for i in range(0,n,2):
                ansatz.rzz(self.new_param('gamma'),i,i+1)
                ansatz.ryy(self.new_param('gamma'),i,i+1)
                ansatz.rxx(self.new_param('gamma'),i,i+1)

        return ansatz