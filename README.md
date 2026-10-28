# Kagome Heisenberg Antiferromagnet with VQE

Variational quantum eigensolver (VQE) study of the spin-1/2 Heisenberg antiferromagnet on the kagome lattice, comparing two variational ansatzes.

This project began as my entry to the **IBM Quantum Open Science Prize 2022**. It was **not a winning entry**; I later developed it into a conference paper:

> A. W. Shaffar, A. A. Hakim, H. Jung, and H. Shin, "Comparative Analysis of Variational Ansatzes in the Kagome Heisenberg Antiferromagnet," *2023 Korean Institute of Communications and Information Sciences (KICS) Fall Conference*, pp. 106–107, 2023.

## The problem

Find the ground state of the Heisenberg Hamiltonian on the kagome lattice,

```
H = Σ_<ij> ( X_i X_j + Y_i Y_j + Z_i Z_j )
```

summed over neighbouring spins. The IBM challenge asked for a relative error below 1% against the exact ground-state energy. Exact solutions become infeasible as the system grows, which is why a variational quantum approach is interesting.

## What I compared

| Ansatz | Idea | Parameters |
|---|---|---|
| **Resonating Valence Bond (RVB)** | Built from eSWAP gates; captures entanglement between qubit pairs | n × l (12 for one layer on 12 qubits) |
| **Hamiltonian Variational Ansatz (HVA)** | Built from XX, YY, and ZZ rotations derived from the Hamiltonian itself | 2 per layer |

Both were optimized with Adam using identical settings (learning rate 0.01). The repository also contains SPSA runs for both ansatzes, which are not reported in the paper.

## Findings (from the paper)

- The HVA starts further from the target energy but converges toward the optimum, even though it has far fewer parameters than the RVB ansatz (2 versus 12).
- The HVA uses a deeper circuit, which could hurt performance on noisy intermediate-scale quantum (NISQ) hardware.
- The paper reports results within 1% of the classically computed ground-state energy, and concludes that the HVA is a good choice for this model.

## Repository contents

| File | What it is |
|---|---|
| `kagome-vqe.ipynb` | Main notebook: runs the VQE experiments and produces the results |
| `heisenberg_model.py` | Kagome Heisenberg Hamiltonian |
| `custom_ansatz.py` | Gate-level RVB and HVA ansatzes |
| `pulse_ansatz.py` | Pulse-level ansatz (exploratory; not part of the paper) |
| `HVA_adam.csv`, `RVB_adam.csv`, `HVA_spsa.csv`, `RVB_spsa.csv` | Optimization histories |
| `kagome_unit_cell.png` | The lattice unit cell used |
| `requirements.txt` | Python dependencies |

## Running it

```bash
pip install -r requirements.txt
jupyter notebook kagome-vqe.ipynb
```

## Limitations

- The results come from **noiseless simulation**. Effects of noise on NISQ devices are not studied, as the paper notes.
- The project did not win the IBM prize.
- Only a small system (12 qubits) is studied.

## Reference

Challenge description: IBM Quantum Open Science Prize 2022 (the original brief).
