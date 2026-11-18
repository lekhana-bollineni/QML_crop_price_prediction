# Hybrid Quantum Machine Learning Methodology

## 1. Mathematical Foundations & Quantum State Encoding

In hybrid quantum-classical machine learning, classical data vectors $\mathbf{x} \in \mathbb{R}^d$ are encoded into parameterized quantum states $|\psi(\mathbf{x})\rangle$ in a $2^n$-dimensional Hilbert space $\mathcal{H}$, where $n$ is the number of qubits.

### A. Angle Feature Encoding
Rather than blindly assigning all dataset features to qubits, feature ranking via Random Forest impurity and mutual information identified the **Top-8 Quantum Features**:
1. `price_current_month` ($x_0$)
2. `price_min_current_month` ($x_1$)
3. `tmin_current_month` ($x_2$)
4. `price_max_current_month` ($x_3$)
5. `price_change_ratio_1m` ($x_4$)
6. `rainfall_total_current_month` ($x_5$)
7. `price_median_current_month` ($x_6$)
8. `sin_month` ($x_7$)

Each feature $x_i$ is mapped into $[0, \pi]$ using training min-max bounds:
$$\theta_i = \pi \cdot \frac{x_i - \min(x_i)}{\max(x_i) - \min(x_i)}$$

Angle embedding is executed using single-qubit Pauli-Y rotations:
$$U_{enc}(\mathbf{x}) = \bigotimes_{i=0}^{n-1} R_y(\theta_i) = \bigotimes_{i=0}^{n-1} \exp\left(-i \frac{\theta_i}{2} Y\right)$$

Initial state:
$$|\psi_0(\mathbf{x})\rangle = U_{enc}(\mathbf{x}) |0\rangle^{\otimes n} = \bigotimes_{i=0}^{n-1} \left(\cos\frac{\theta_i}{2} |0\rangle + \sin\frac{\theta_i}{2} |1\rangle\right)$$

---

## 2. Hybrid Quantum Architectures

### 1. Variational Quantum Regressor (VQR)
- **Qubit Register**: 8 qubits ($n=8$).
- **Ansatz**: Multi-layer parameterized entanglement:
  $$U(\boldsymbol{\theta}) = \prod_{l=1}^{L} \left[ \left(\prod_{i=0}^{n-1} \text{CNOT}_{i, (i+1)\%n}\right) \left(\bigotimes_{i=0}^{n-1} R_y(\theta_{l, i})\right) \right]$$
- **Measurement Readout**: Pauli-Z expectation value on anchor wire:
  $$\langle Z_0 \rangle = \langle 0 | U_{enc}^\dagger U^\dagger(\boldsymbol{\theta}) Z_0 U(\boldsymbol{\theta}) U_{enc} | 0 \rangle \in [-1, 1]$$
- **Classical Trainable Affine Head**:
  $$\hat{y}_{norm} = w \cdot \langle Z_0 \rangle + b$$
- **Optimization**: PyTorch Adam optimizer with MSE loss and adjoint differentiation on PennyLane's `lightning.qubit`.

### 2. Quantum Neural Network Regressor (QNN)
- **Qubit Register**: 8 qubits ($n=8$).
- **Quantum Feature Layer**: Generates an 8-dimensional expectation state vector:
  $$\mathbf{z}(\mathbf{x}) = \left[ \langle Z_0 \rangle, \langle Z_1 \rangle, \dots, \langle Z_{n-1} \rangle \right]^T \in [-1, 1]^n$$
- **Classical Dense Network Head**:
  $$\mathbf{h} = \text{ReLU}(W_1 \mathbf{z}(\mathbf{x}) + \mathbf{b}_1), \quad W_1 \in \mathbb{R}^{16 \times 8}$$
  $$\hat{y}_{norm} = \mathbf{w}_2^T \mathbf{h} + b_2, \quad \mathbf{w}_2 \in \mathbb{R}^{16}$$
- **End-to-End Gradients**: Gradients backpropagate through classical dense weights and quantum rotation angles simultaneously via PennyLane's `TorchLayer`.

### 3. Quantum Kernel Support Vector Regression (QSVR)
- **Quantum Feature Map**: State-preparation unitaries incorporating entangling CNOT ring topology and Phase rotations:
  $$|\psi(\mathbf{x})\rangle = U_Z(\mathbf{x}) \cdot \text{CNOT}_{ring} \cdot U_Y(\mathbf{x}) |0\rangle^{\otimes 8}$$
- **Quantum Kernel Entry**: The kernel represents the transition fidelity between quantum states:
  $$K(\mathbf{x}_i, \mathbf{x}_j) = |\langle \psi(\mathbf{x}_i) | \psi(\mathbf{x}_j) \rangle|^2$$
- **State-Vector Acceleration**: By computing state vectors $|\psi(\mathbf{x}_i)\rangle \in \mathbb{C}^{256}$ once per observation, the $N \times N$ Gram matrix is computed via BLAS matrix multiplication:
  $$K_{ij} = |(V V^\dagger)_{ij}|^2$$
  reducing quantum circuit evaluations from $O(N^2)$ to $O(N)$.
- **SVR Optimization**: Solves the dual convex quadratic program:
  $$\max_{\boldsymbol{\alpha}, \boldsymbol{\alpha}^*} -\frac{1}{2} (\boldsymbol{\alpha} - \boldsymbol{\alpha}^*)^T K (\boldsymbol{\alpha} - \boldsymbol{\alpha}^*) - \epsilon \sum_{i=1}^N (\alpha_i + \alpha_i^*) + \sum_{i=1}^N y_i (\alpha_i - \alpha_i^*)$$
  subject to $\sum_{i=1}^N (\alpha_i - \alpha_i^*) = 0$ and $0 \le \alpha_i, \alpha_i^* \le C$.

---

## 3. Computational Budget & Scientific Accounting
- **Training Budget**: Controlled sample size $N_{train} = 5,000$ observations to model realistic near-term quantum simulator budgets.
- **Test Evaluation**: Evaluated across all $N_{test} = 5,109$ observations of the 2024 test partition.
- **Empirical Findings**: QSVR achieved a competitive test MAPE of **16.31%** (closely matching XGBoost's 17.30% and LightGBM's 17.82%), demonstrating high feature expressiveness in reproducing nonlinear market states.
