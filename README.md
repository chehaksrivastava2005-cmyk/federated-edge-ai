# Federated Edge AI — Smart Infrastructure Platform

**Secure and Communication-Efficient Federated Edge Intelligence for Privacy-Preserving Smart Infrastructure Systems**

A research-grade federated learning framework where distributed edge devices collaboratively train AI models without sharing raw data. Built for smart city infrastructure across three real-world applications: traffic monitoring, industrial fault detection, and surveillance.

---

## Overview

Traditional machine learning requires centralising all data on a single server — a critical privacy and security risk for smart infrastructure. This system implements **Hierarchical Federated Learning** where each edge device (camera, sensor, gateway) trains locally on its private data and only transmits compressed model updates to a central aggregator.

The system introduces four core novelty contributions:

1. **Hierarchical FL Orchestration** — 3-tier architecture (Edge → Regional Aggregator → Global Coordinator) for city-scale deployment
2. **Self-Healing Byzantine Security** — automatic detection and isolation of malicious nodes with trust-based reintegration
3. **Communication Intelligence Engine** — adaptive Top-K sparsification + INT8 quantization reducing bandwidth by ~70%
4. **Context-Aware Client Participation** — dynamic participation based on device reliability, trust score, and network quality

---

## Results

| Application | FL Accuracy | Centralized Baseline | Gap | Comm. Saved |
|---|---|---|---|---|
| Smart Traffic | **97.62%** | 97.08% | +0.54% | ~70% |
| Industrial Monitoring | **48.50%** | 48.96% | −0.46% | ~70% |
| Smart Surveillance | **95.67%** | 98.75% | −3.08% | ~70% |

- **Attack detection rate:** 100% (Byzantine client caught from round 2 onward)
- **Communication reduction:** ~70% per round via gradient sparsification + INT8 quantization
- **Self-healing:** Malicious nodes auto-isolated at trust = 0, reintegrated after recovery

---

## Project Structure

```
federated_edge_ai/
│
├── main.py                        ← Entry point — runs full FL pipeline
├── dashboard.html                 ← Interactive web dashboard (open in browser)
├── requirements.txt               ← Python dependencies
│
├── server/
│   └── federated_server.py        ← FedAvg aggregation, model distribution
│
├── clients/
│   └── edge_client.py             ← Local training, gradient compression
│
├── models/
│   ├── traffic_model.py           ← MLP classifier for traffic conditions
│   ├── industrial_model.py        ← MLP regressor for RUL prediction
│   └── surveillance_model.py      ← MLP classifier for scene alerts
│
├── datasets/
│   └── data_loader.py             ← Load, preprocess, Non-IID partition
│
├── security/
│   └── anomaly_detector.py        ← Z-score + cosine similarity detection
│
├── optimization/
│   └── compressor.py              ← Top-K sparsification + INT8 quantization
│
├── utils/
│   ├── config.py                  ← Hyperparameter configuration
│   └── metrics.py                 ← Round-by-round metrics tracker
│
├── visualizations/
│   └── plotter.py                 ← Matplotlib result graphs
│
├── results/                       ← JSON experiment outputs (auto-generated)
└── visualizations/                ← PNG graphs (auto-generated)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Framework | PyTorch 2.x |
| Federated Learning | Custom FedAvg implementation |
| Data Processing | NumPy, Pandas, Scikit-learn |
| Visualization (backend) | Matplotlib |
| Dashboard (frontend) | HTML5 Canvas, vanilla JS |
| Language | Python 3.9+ |

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/federated-edge-ai.git
cd federated-edge-ai
```

### 2. Install dependencies

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

### 3. Run quick demo (recommended first run)

```bash
python main.py --demo
```

Runs 5 rounds with 3 clients across all three applications. Completes in under 2 minutes. Verifies the full pipeline works end-to-end.

### 4. Run full experiment

```bash
python main.py --app all --rounds 10 --clients 5
```

### 5. Run individual applications

```bash
python main.py --app traffic      --rounds 10 --clients 5
python main.py --app industrial   --rounds 10 --clients 5
python main.py --app surveillance --rounds 10 --clients 5
```

### 6. Run baseline comparison (no security, no compression)

```bash
python main.py --app all --rounds 10 --clients 5 --no-security --no-compression
```

### 7. Open the dashboard

Double-click `dashboard.html` or run:

```bash
# Windows
start dashboard.html

# Mac
open dashboard.html

# Linux
xdg-open dashboard.html
```

---

## Command Reference

| Flag | Default | Description |
|---|---|---|
| `--app` | `all` | Application: `traffic`, `industrial`, `surveillance`, `all` |
| `--rounds` | `10` | Number of FL aggregation rounds |
| `--clients` | `5` | Number of simulated edge devices |
| `--demo` | `false` | Quick demo mode (5 rounds, 3 clients) |
| `--no-security` | `false` | Disable anomaly detection (baseline) |
| `--no-compression` | `false` | Disable gradient compression (baseline) |

---

## Datasets

The system runs fully on synthetic data by default. Real datasets can be plugged in by placing files in `datasets/raw/`.

### NASA CMAPSS — Industrial (Recommended, 2 MB)

Download from the [NASA PCOE Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/).

```
datasets/raw/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

### Metro Interstate Traffic (Kaggle, ~10 MB)

Download from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/traffic-volume-data-set).

```
datasets/raw/traffic/
└── Metro_Interstate_Traffic_Volume.csv
```

### MS-COCO 2017 val — Surveillance (~1 GB)

```bash
# Download annotations only (241 MB — images not required)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d datasets/raw/coco/

# Extract features (run once)
python datasets/surveillance_data.py
```

---

## System Architecture

```
Edge Devices (Layer 3)
      │  local training on private data
      │  compressed gradient updates only
      ▼
Regional Aggregators (Layer 2)   ×4 regions
      │  pre-aggregation reduces global comm cost ~60%
      │  filters malicious updates per region
      ▼
Global Coordinator (Layer 1)
      │  FedAvg across all regional models
      │  redistributes global model
      ▼
Global Model
```

### Privacy Guarantee

Raw data never leaves the edge device. Only model weight deltas (gradients) are transmitted. Even these are:
- Sparsified to top 30% by magnitude
- Quantized from float32 → int8 (4× byte reduction)
- Inspected for Byzantine anomalies before aggregation

---

## Security Module

The system simulates and detects **Byzantine attacks** — compromised edge devices that submit deliberately corrupted model updates.

**Detection methods:**

| Method | How it works |
|---|---|
| Norm-based Z-score | Compute L2 norm of each client's gradient delta. Reject if \|Z\| > 2.5σ above median |
| Cosine similarity | Sign-flipped updates point in the opposite direction. Reject if cosine similarity < −0.5 |
| Trust scoring | Each client starts at 1.0. Flagged: −0.3. Honest rounds: +0.05. Excluded below 0.3 |
| Self-healing | Auto-isolate at trust = 0. Reintegrate after recovery with probation score of 0.25 |

---

## Communication Optimization

| Technique | Reduction | Accuracy Impact |
|---|---|---|
| Top-30% sparsification | −70% parameters | < 1% |
| INT8 quantization | 4× bytes | < 0.5% |
| Combined | **~87% total** | < 1.5% |

---

## Output Files

After running an experiment:

```
results/
├── traffic_results.json        ← Round-by-round accuracy, loss, comm cost
├── industrial_results.json
└── surveillance_results.json

visualizations/
├── accuracy_vs_rounds.png      ← FL accuracy per round vs centralized baseline
├── communication_cost.png      ← Compressed vs uncompressed bandwidth
├── security_detection.png      ← Trust score decay for malicious clients
├── loss_curves.png             ← Training loss across all applications
├── federated_vs_centralized.png← Final accuracy comparison bar chart
└── dashboard.png               ← Summary dashboard
```

---

## Dashboard Features

The interactive `dashboard.html` requires no server or build step — just open in any browser.

| Page | Contents |
|---|---|
| Dashboard | Live city digital twin, KPI cards, trust scores, activity feed, scenario controls |
| Analytics | Accuracy/loss charts, regional comparison, communication savings |
| Security | Threat heatmap, Byzantine detection, active threat list |
| FL Network | 3-tier hierarchy visualization, research results, novelty contributions |

**Demo scenarios (bottom control bar):**
- `▶ Start FL Training` — watch accuracy climb in real time
- `☠ Byzantine Attack` — inject a malicious node mid-training
- `Traffic Surge` — spike North District load
- `Node Failure` — test self-healing reconnection
- `Multi-Threat` — simultaneous attack + failure + congestion

---

## Research Contributions

This project contributes to the following research areas:

- **Federated Learning at Scale** — hierarchical aggregation for city-level deployment
- **Byzantine Robustness** — self-healing trust system with automatic node recovery  
- **Communication Efficiency** — combined sparsification and quantization pipeline
- **Privacy-Preserving AI** — end-to-end system with no raw data transmission
- **Smart Infrastructure** — applied FL to three real infrastructure domains

---

## References

| Paper | Relevance |
|---|---|
| McMahan et al. (2017) — *Communication-Efficient Learning of Deep Networks from Decentralized Data* | FedAvg algorithm |
| Lin et al. (2017) — *Deep Gradient Compression* | Gradient sparsification |
| Blanchard et al. (2017) — *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent* | Byzantine detection |
| Konečný et al. (2016) — *Federated Learning: Strategies for Improving Communication Efficiency* | Quantization |

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

## Authors

Developed as a final year / semester research project.  
Department of Computer Science — Federated Edge AI Team.
