"""The pulse-based ansatz module"""
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

from qiskit import pulse
from qiskit.circuit import Gate, QuantumCircuit, Parameter

class AnsatzBuilder:
    """A class to help build circuit Ansatze.

    This class defines several methods that add the pulse gates that we are interested in.
    The builder also creates the template schedules along with these gates. Currently, the
    builder supports the following gates:

    - cr: A single cross-resonance tone on a gate.

    The builder also returns a parameter wrapper config that defines the functions that connect
    the values of the parameters being optimized and those assigned in the pulse schedules.
    
    see: https://arxiv.org/abs/2303.02410
    """

    def __init__(
        self,
        backend,
        physical_qubits: tuple,
        frequency_shift_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
        max_duration: float = 800,
    ):
        """Initialize the class

        Args:
            backend: The backend for which to construct the ansatz. This is needed so that the
                pulse builder knows on which channels it should apply the pulses. For simulator
                backends the pulses will be applied on control channels that are made up on the
                fly.
            physical_qubits: A tuple mapping the program qubits (indices) to the physical
                qubits (values) to which the circuit will be transpiled. This is needed because
                the pulse schedules are built for physical qubits.
            frequency_shift_ranges: The allowed range of frequency shifts for CCR gates.
            max_duration: The maximum allowed duration for any pulse duration parameter.
        """
        self._params = []
        self._backend = backend
        self._physical_qubits_map = physical_qubits
        self._control_channels = {}
        self._wrapper_config = {}
        self._max_duration = max_duration
        self._max_freq_shift = frequency_shift_ranges

    @property
    def backend(self):
        """Return the backend of th builder."""
        return self._backend

    @property
    def wrapper_config(self) -> Dict:
        """Return the config for the parameter wrapping."""
        return self._wrapper_config

    def _get_control_channel(self, qubits: Tuple[int, ...]) -> pulse.ControlChannel:
        """Get a control channel and if the backend does not provide one then we make one up."""

        # Try except needed because of BackendV1 vs V2
        if hasattr(self._backend, "control_channel"):
            return self._backend.control_channel(qubits)[0]
        elif hasattr(self._backend, "configuration"):
            if hasattr(self._backend.configuration(), "control_channels"):
                if qubits in self._backend.configuration().control_channels:
                    return self._backend.configuration().control_channels[qubits][0]

        # Use a self-made control channel. Only simulators should reach this point.
        if qubits not in self._control_channels:
            max_chan = len(self._control_channels)
            self._control_channels[qubits] = pulse.ControlChannel(max_chan + 1)

        return self._control_channels[qubits]

    def new_param(self, paramstr = 'p') -> Parameter:
        """Create a new parameter and save it."""
        param = Parameter("".join([paramstr, str(len(self._params))]))
        self._params.append(param)
        return param

    def _get_schedule(self, name: str, qubits: tuple) -> pulse.ScheduleBlock:
        """Return a schedule block from the backend."""
        return self._backend.defaults().instruction_schedule_map.get(name, qubits)

    def add_rx(self, circuit: QuantumCircuit, qubit: int):
        """Create an x gate with a parametric amplitude and attach a pulse to it."""
        param = self.new_param('phi')

        circuit.rx(param, qubit)

    def add_cr(
            self,
            circuit: QuantumCircuit,
            qubits: Tuple[int, int],
            duration: Optional[Parameter] = None
    ):
        """Add a cross-resonance gate to the circuit.

        The parameters are duration and amplitude in this order.

        Args:
            circuit: The circuit to which to append the ccr gate.
            qubits: The program qubits.
            duration: An optional duration parameter. If None is given (the default)
                then a new parameter will be created.
        """
        if duration is None:
            duration = self.new_param('t')

        circuit.append(Gate("cr", 2, params=[duration, self.new_param('phi')]), qubits)

    def add_schedules(self, circuit: QuantumCircuit):
        """Parse the quantum circuit and add the schedules to it."""
        qreg = circuit.qregs[0]

        for inst in circuit.data:
            if inst.operation.name in ["ccr", "cr", "rx"]:
                params = inst.operation.params
                qubits = tuple(qreg.index(qubit) for qubit in inst.qubits)
                pqubits = tuple(self._physical_qubits_map[qubit] for qubit in qubits)

                if inst.operation.name == "cr":
                    self.add_cr_schedule(params, pqubits, circuit)

                if inst.operation.name == "rx":
                    self.add_rx_schedule(params, pqubits, circuit)

    @staticmethod
    def _param_idx(circuit: QuantumCircuit, param: Parameter):
        """Get the index of the parameter."""
        for idx, circ_param in enumerate(circuit.parameters):
            if circ_param == param:
                return idx

        raise ValueError(f"Parameter {param} not found in {circuit.parameters}.")

    def add_rx_schedule(self, params: List[Parameter], pqubits: tuple, circuit: QuantumCircuit):
        """Add the schedule for the rx gate to the circuit.

        Args:
            params: The parameters of the rx gate. This should be a simple amplitude of the
                pulse.
            pqubits: The physical qubit that the pulse schedule will run on.
            circuit: The circuit to which to add the calibrations.
        """
        # Parameter wrapper for the amplitude of the pulse. If it is a float do not add a wrapper.
        try:
            amp = float(params[0])
        except TypeError:
            amp = params[0]
            self._wrapper_config[self._param_idx(circuit, params[0])] = (
                "SinWrapper", (1, 1, 0, 0), f"amp[{params[0].name}]",
            )

        x_pulse = self._get_schedule("x", tuple(pqubits)).instructions[0][1].pulse

        with pulse.build(name="rx", backend=self._backend) as x_sched:
            pulse.play(
                pulse.Drag(x_pulse.duration, amp=amp, sigma=x_pulse.sigma, beta=x_pulse.beta),
                pulse.drive_channel(pqubits[0])
            )

        circuit.add_calibration("rx", qubits=tuple(pqubits), schedule=x_sched, params=[amp])

    def add_cr_schedule(self, params: List[Parameter], pqubits: tuple, circuit: QuantumCircuit):
        """Create the cross-resonance schedule and add it to the circuit.

        Args:
            params: The parameters of the cr gate. This should be the duration and the
                amplitude in that same order.
            pqubits: The physical qubit that the pulse schedule will run on.
            circuit: The circuit to which to add the calibrations.
        """
        # Duration parameter wrapper. If it can be converted to float do not add a wrapper.
        try:
            duration = int(params[0])
        except TypeError:
            duration = params[0]
            self._wrapper_config[self._param_idx(circuit, params[0])] = (
                "SinDurationWrapper",
                (self._max_duration / 2, 1, self._max_duration / 2 + 4 * 64, 160),
                f"dur[{params[0].name}]",
            )

        # Amplitude parameter wrapper. If it can be converted to float do not add a wrapper.
        try:
            amp = float(params[1])
        except TypeError:
            amp = params[1]
            self._wrapper_config[self._param_idx(circuit, params[1])] = (
                "SinWrapper", (1, 1, 0, 0), f"amp[{params[1].name}]"
            )

        with pulse.build(backend=self._backend) as cr_sched:
            u_chan = self._get_control_channel(pqubits)
            pulse.play(
                pulse.GaussianSquare(
                    duration=duration, amp=amp, sigma=64, risefall_sigma_ratio=2
                ),
                u_chan,
            )

        circuit.add_calibration("cr", qubits=pqubits, schedule=cr_sched, params=[duration, amp])


class PulseAnsatz(ABC):
    """A base class for pulse Ansatz creation."""

    def __init__(
        self,
        num_qubits: int,
        builder: AnsatzBuilder,
        reps: int = 1,
        entanglement: Union[str, List[Tuple]] = "interleaved",
        add_rz: bool = False,
        final_rx: bool = True,
    ):
        """Initialize the ansatz creator.

        Args:
            num_qubits: The number of qubits for which to generate the circuit. The physical
                qubits that will be used must be given in the AnsatzBuilder so that it can
                create the schedules.
            builder: The ansatz builder that helps manage parameters, parameter wrappers,
                and schedules.
            entanglement: The type of entanglement structure to use. For now, we support
                interleaved, and a list of tuple the specify on which qubits to apply the
                gates.
            add_rz: A boolean to indicate if virtual-Z gates should be added to the variational
                form. By default this value is False.
            final_rx: A boolean to indicate if a final layer of Rx rotations should be added
                to the variational form. By default, this value is True.
        """
        self._builder = builder
        self._num_qubits = num_qubits
        self._reps = reps
        self._entanglement = entanglement
        self._add_rz = add_rz
        self._final_rx = final_rx

    def circuit(self) -> QuantumCircuit:
        """Make the quantum circuit."""
        ansatz = QuantumCircuit(self._num_qubits)

        for _ in range(self._reps):
            ansatz.compose(self._layer(), inplace=True)

        if self._final_rx:
            for qubit in range(self._num_qubits):
                self._builder.add_rx(ansatz, qubit)

        return ansatz

    @abstractmethod
    def _layer(self) -> QuantumCircuit:
        """Creates a single layer of the variational form."""


class CRAnsatz(PulseAnsatz):
    """A quantum circuit variational form with cross-resonance gates."""

    def _layer(self) -> QuantumCircuit:
        """Create a single-layer of the pulse variational form."""
        ansatz = QuantumCircuit(self._num_qubits)

        for qubit in range(self._num_qubits):
            self._builder.add_rx(ansatz, qubit)

            if self._add_rz:
                ansatz.rz(self._builder.new_param('phi'), qubit)

        if self._entanglement == "interleaved":
            for qubit in range(0, self._num_qubits, 2):
                self._builder.add_cr(ansatz, (qubit, qubit+1))

            for qubit in range(1, self._num_qubits-1, 2):
                self._builder.add_cr(ansatz, (qubit, qubit+1))

        if isinstance(self._entanglement, List):
            for pair in self._entanglement:
                self._builder.add_cr(ansatz, pair)

        return ansatz