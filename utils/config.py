# utils/config.py
"""
Configuration dataclass for all federated learning hyperparameters.
Centralizing config means you change ONE file to run different experiments.
"""
from dataclasses import dataclass, asdict

@dataclass
class Config:
    # --- Federated Learning ---
    num_rounds: int = 10        # How many global aggregation rounds
    num_clients: int = 5        # Number of simulated edge devices
    local_epochs: int = 2       # Epochs each client trains per round
    learning_rate: float = 0.01
    batch_size: int = 32
    fraction_fit: float = 1.0   # Fraction of clients selected per round

    # --- Security ---
    inject_malicious: bool = True   # Inject 1 malicious client for demo
    anomaly_threshold: float = 2.5  # Z-score threshold for rejection
    min_trust_score: float = 0.3    # Below this → client excluded

    # --- Communication Optimization ---
    compression_ratio: float = 0.3  # Keep top 30% gradients (sparsification)
    quantize_bits: int = 8          # Quantize to 8-bit integers

    # --- Reproducibility ---
    random_seed: int = 42

    def to_dict(self):
        return asdict(self)