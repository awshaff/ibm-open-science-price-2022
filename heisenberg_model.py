# Modifed from the Ising model found here: https://qiskit.org/documentation/nature/_modules/qiskit_nature/problems/second_quantization/lattice/models/ising_model.html#IsingModel

"""The Heisenberg model"""
import logging
import numpy as np
from fractions import Fraction
from typing import Optional

from qiskit_nature.operators.second_quantization import SpinOp
from qiskit_nature.problems.second_quantization.lattice.lattices import Lattice
from qiskit_nature.problems.second_quantization.lattice.models.lattice_model import LatticeModel

class HeisenbergModel(LatticeModel):
    """The Heisenberg model."""

    def coupling_matrix(self) -> np.ndarray:
        """Return the coupling matrix."""
        return self.interaction_matrix()


    @classmethod
    def uniform_parameters(
        cls,
        lattice: Lattice,
        uniform_interaction: complex,
        uniform_onsite_potential: complex,
    ) -> "HeisenbergModel":
        """Set a uniform interaction parameter and on-site potential over the input lattice.

        Args:
            lattice: Lattice on which the model is defined.
            uniform_interaction: The interaction parameter.
            uniform_onsite_potential: The on-site potential.

        Returns:
            The Lattice model with uniform parameters.
        """
        return cls(
            cls._generate_lattice_from_uniform_parameters(
                lattice, uniform_interaction, uniform_onsite_potential
            )
        )


    @classmethod
    def from_parameters(
        cls,
        interaction_matrix: np.ndarray,
    ) -> "HeisenbergModel":
        """Return the Hamiltonian of the Lattice model
        from the given interaction matrix and on-site interaction.

        Args:
            interaction_matrix: A real or complex valued square matrix.

        Returns:
            LatticeModel: The Lattice model generated from the given interaction
                matrix and on-site interaction.

        Raises:
            ValueError: If the interaction matrix is not square matrix, it is invalid.
        """
        return cls(cls._generate_lattice_from_parameters(interaction_matrix))


    def second_q_ops(self, display_format: Optional[str] = None) -> SpinOp:
        """Return the Hamiltonian of the Heisenberg model in terms of `SpinOp`.

        Args:
            display_format: Not supported for Spin operators. If specified, it will be ignored.

        Returns:
            SpinOp: The Hamiltonian of the Heisenberg model.
        """
        if display_format is not None:
            logger.warning(
                "Spin operators do not support display-format. Provided display-format "
                "parameter will be ignored."
            )
        ham = []
        weighted_edge_list = self._lattice.weighted_edge_list
        register_length = self._lattice.num_nodes
        # kinetic terms
        for node_a, node_b, weight in weighted_edge_list:
            if node_a == node_b:
                index = node_a
                ham.append((f"X_{index}", weight))

            else:
                index_left = node_a
                index_right = node_b
                coupling_parameter = weight
                ham.append((f"X_{index_left} X_{index_right}", coupling_parameter))
                ham.append((f"Y_{index_left} Y_{index_right}", coupling_parameter))
                ham.append((f"Z_{index_left} Z_{index_right}", coupling_parameter))
        return SpinOp(ham, spin=Fraction(1, 2), register_length=register_length)