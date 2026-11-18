"""
Quantum Machine Learning (QML) Architectures for Crop Price Prediction.
Implements:
1. VariationalQuantumRegressor (VQR)
2. QuantumNeuralNetworkRegressor (QNN)
3. QuantumKernelSVR (QSVR)
Using PennyLane (lightning.qubit simulator) and PyTorch / Scikit-Learn.
"""
import time
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import SVR
import pennylane as qml

# Determine optimal device
def get_qml_device(wires):
    try:
        return qml.device('lightning.qubit', wires=wires)
    except Exception:
        return qml.device('default.qubit', wires=wires)

# ==============================================================================
# 1. VARIATIONAL QUANTUM REGRESSOR (VQR)
# ==============================================================================
class VQRModule(nn.Module):
    def __init__(self, n_qubits=8, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        dev = get_qml_device(n_qubits)
        
        @qml.qnode(dev, interface='torch', diff_method='adjoint')
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))
            
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        q_out = self.qlayer(x)
        return (self.scale * q_out + self.bias).view(-1)

class VariationalQuantumRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_qubits=8, n_layers=2, lr=0.02, epochs=5, batch_size=64, sample_size=5000, random_state=42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.random_state = random_state
        self.model = None
        self.y_mean_ = 0.0
        self.y_std_ = 1.0
        self.training_time_ = 0.0
        self.training_samples_ = 0

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        
        # Subsample if requested
        if len(X_arr) > self.sample_size:
            idx = np.random.choice(len(X_arr), size=self.sample_size, replace=False)
            X_train = X_arr[idx]
            y_train = y_arr[idx]
        else:
            X_train = X_arr
            y_train = y_arr
            
        self.training_samples_ = len(X_train)
        self.y_mean_ = float(np.mean(y_train))
        self.y_std_ = float(np.std(y_train)) if np.std(y_train) > 1e-6 else 1.0
        y_norm = (y_train - self.y_mean_) / self.y_std_
        
        dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_norm))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = VQRModule(n_qubits=self.n_qubits, n_layers=self.n_layers)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        start_time = time.perf_counter()
        self.model.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                
        self.training_time_ = time.perf_counter() - start_time
        return self

    def predict(self, X):
        self.model.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        loader = DataLoader(TensorDataset(torch.from_numpy(X_arr)), batch_size=self.batch_size * 2, shuffle=False)
        preds = []
        with torch.no_grad():
            for (bx,) in loader:
                p = self.model(bx).detach().cpu().numpy().reshape(-1)
                preds.append(p)
        pred_norm = np.concatenate(preds)
        return pred_norm * self.y_std_ + self.y_mean_

    def save(self, file_path):
        state = {
            'state_dict': self.model.state_dict(),
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'lr': self.lr,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'sample_size': self.sample_size,
            'y_mean_': self.y_mean_,
            'y_std_': self.y_std_,
            'training_time_': self.training_time_,
            'training_samples_': self.training_samples_
        }
        joblib.dump(state, file_path)

    @classmethod
    def load(cls, file_path):
        state = joblib.load(file_path)
        inst = cls(
            n_qubits=state['n_qubits'],
            n_layers=state['n_layers'],
            lr=state['lr'],
            epochs=state['epochs'],
            batch_size=state['batch_size'],
            sample_size=state['sample_size']
        )
        inst.y_mean_ = state['y_mean_']
        inst.y_std_ = state['y_std_']
        inst.training_time_ = state['training_time_']
        inst.training_samples_ = state['training_samples_']
        inst.model = VQRModule(n_qubits=inst.n_qubits, n_layers=inst.n_layers)
        inst.model.load_state_dict(state['state_dict'])
        inst.model.eval()
        return inst

# ==============================================================================
# 2. QUANTUM NEURAL NETWORK REGRESSOR (QNN)
# ==============================================================================
class QNNModule(nn.Module):
    def __init__(self, n_qubits=8, n_layers=1, hidden_dim=16):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        dev = get_qml_device(n_qubits)
        
        @qml.qnode(dev, interface='torch', diff_method='adjoint')
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.head = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        q_features = self.qlayer(x)
        return self.head(q_features).view(-1)

class QuantumNeuralNetworkRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_qubits=8, n_layers=1, hidden_dim=16, lr=0.01, epochs=5, batch_size=64, sample_size=5000, random_state=42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.random_state = random_state
        self.model = None
        self.y_mean_ = 0.0
        self.y_std_ = 1.0
        self.training_time_ = 0.0
        self.training_samples_ = 0

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        
        if len(X_arr) > self.sample_size:
            idx = np.random.choice(len(X_arr), size=self.sample_size, replace=False)
            X_train = X_arr[idx]
            y_train = y_arr[idx]
        else:
            X_train = X_arr
            y_train = y_arr
            
        self.training_samples_ = len(X_train)
        self.y_mean_ = float(np.mean(y_train))
        self.y_std_ = float(np.std(y_train)) if np.std(y_train) > 1e-6 else 1.0
        y_norm = (y_train - self.y_mean_) / self.y_std_
        
        dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_norm))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = QNNModule(n_qubits=self.n_qubits, n_layers=self.n_layers, hidden_dim=self.hidden_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        start_time = time.perf_counter()
        self.model.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                
        self.training_time_ = time.perf_counter() - start_time
        return self

    def predict(self, X):
        self.model.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        loader = DataLoader(TensorDataset(torch.from_numpy(X_arr)), batch_size=self.batch_size * 2, shuffle=False)
        preds = []
        with torch.no_grad():
            for (bx,) in loader:
                p = self.model(bx).detach().cpu().numpy().reshape(-1)
                preds.append(p)
        pred_norm = np.concatenate(preds)
        return pred_norm * self.y_std_ + self.y_mean_

    def save(self, file_path):
        state = {
            'state_dict': self.model.state_dict(),
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'hidden_dim': self.hidden_dim,
            'lr': self.lr,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'sample_size': self.sample_size,
            'y_mean_': self.y_mean_,
            'y_std_': self.y_std_,
            'training_time_': self.training_time_,
            'training_samples_': self.training_samples_
        }
        joblib.dump(state, file_path)

    @classmethod
    def load(cls, file_path):
        state = joblib.load(file_path)
        inst = cls(
            n_qubits=state['n_qubits'],
            n_layers=state['n_layers'],
            hidden_dim=state['hidden_dim'],
            lr=state['lr'],
            epochs=state['epochs'],
            batch_size=state['batch_size'],
            sample_size=state['sample_size']
        )
        inst.y_mean_ = state['y_mean_']
        inst.y_std_ = state['y_std_']
        inst.training_time_ = state['training_time_']
        inst.training_samples_ = state['training_samples_']
        inst.model = QNNModule(n_qubits=inst.n_qubits, n_layers=inst.n_layers, hidden_dim=inst.hidden_dim)
        inst.model.load_state_dict(state['state_dict'])
        inst.model.eval()
        return inst

# ==============================================================================
# 3. QUANTUM KERNEL SUPPORT VECTOR REGRESSION (QSVR)
# ==============================================================================
class QuantumKernelSVR(BaseEstimator, RegressorMixin):
    def __init__(self, n_qubits=8, C=10.0, epsilon=0.1, sample_size=5000, random_state=42):
        self.n_qubits = n_qubits
        self.C = C
        self.epsilon = epsilon
        self.sample_size = sample_size
        self.random_state = random_state
        self.svr = None
        self.train_states_ = None
        self.y_mean_ = 0.0
        self.y_std_ = 1.0
        self.training_time_ = 0.0
        self.training_samples_ = 0
        self.dev = get_qml_device(n_qubits)
        self._init_circuit()

    def _init_circuit(self):
        n_q = self.n_qubits
        dev = self.dev
        @qml.qnode(dev)
        def state_circuit(x):
            qml.AngleEmbedding(x, wires=range(n_q), rotation='Y')
            for i in range(n_q - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.AngleEmbedding(x, wires=range(n_q), rotation='Z')
            return qml.state()
        self._state_circuit = state_circuit

    def _compute_states(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        states = []
        for x in X_arr:
            s = self._state_circuit(x)
            states.append(s)
        return np.asarray(states, dtype=np.complex128)

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        
        if len(X_arr) > self.sample_size:
            idx = np.random.choice(len(X_arr), size=self.sample_size, replace=False)
            X_train = X_arr[idx]
            y_train = y_arr[idx]
        else:
            X_train = X_arr
            y_train = y_arr
            
        self.training_samples_ = len(X_train)
        self.y_mean_ = float(np.mean(y_train))
        self.y_std_ = float(np.std(y_train)) if np.std(y_train) > 1e-6 else 1.0
        y_norm = (y_train - self.y_mean_) / self.y_std_
        
        start_time = time.perf_counter()
        # Compute quantum state vectors
        self.train_states_ = self._compute_states(X_train)
        # Compute Quantum Kernel Gram Matrix: K_ij = |<psi_i | psi_j>|^2
        K_train = np.abs(np.dot(self.train_states_, self.train_states_.conj().T))**2
        
        self.svr = SVR(kernel='precomputed', C=self.C, epsilon=self.epsilon)
        self.svr.fit(K_train, y_norm)
        self.training_time_ = time.perf_counter() - start_time
        return self

    def predict(self, X):
        if self._state_circuit is None:
            self._init_circuit()
        test_states = self._compute_states(X)
        # Compute Kernel between test and train states: K_test,train = |V_test * V_train^H|^2
        K_test_train = np.abs(np.dot(test_states, self.train_states_.conj().T))**2
        pred_norm = np.atleast_1d(self.svr.predict(K_test_train)).reshape(-1)
        return pred_norm * self.y_std_ + self.y_mean_

    def save(self, file_path):
        state = {
            'svr': self.svr,
            'train_states_': self.train_states_,
            'n_qubits': self.n_qubits,
            'C': self.C,
            'epsilon': self.epsilon,
            'sample_size': self.sample_size,
            'y_mean_': self.y_mean_,
            'y_std_': self.y_std_,
            'training_time_': self.training_time_,
            'training_samples_': self.training_samples_
        }
        joblib.dump(state, file_path)

    @classmethod
    def load(cls, file_path):
        state = joblib.load(file_path)
        inst = cls(
            n_qubits=state['n_qubits'],
            C=state['C'],
            epsilon=state['epsilon'],
            sample_size=state['sample_size']
        )
        inst.svr = state['svr']
        inst.train_states_ = state['train_states_']
        inst.y_mean_ = state['y_mean_']
        inst.y_std_ = state['y_std_']
        inst.training_time_ = state['training_time_']
        inst.training_samples_ = state['training_samples_']
        inst._init_circuit()
        return inst
