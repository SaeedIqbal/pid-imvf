# PID-IMVF: Mitigating Modality Collapse in Long-Tail Multi-View Learning via Partial Information Decomposition

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official PyTorch implementation of **Identifiable Synergistic Multi-View Learning via Partial Information Decomposition and Nonlinear ICA**. This repository provides the code, configurations, and evaluation scripts to reproduce the results presented in our manuscript.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Issues Resolved & Importance](#-issues-resolved--importance)
3. [Methodology Summary](#-methodology-summary)
4. [Experimental Results](#-experimental-results)
5. [Dataset Details](#-dataset-details)
6. [Installation & Quick Start](#-installation--quick-start)
7. [Citation](#-citation)

---

## 📖 Project Overview
Multi-view learning aims to fuse heterogeneous data sources by capturing both **consistency** (shared information) and **complementarity** (view-unique information). State-of-the-art methods predominantly rely on heuristic architectural decoupling (e.g., separate CNN and attention branches) without information-theoretic guarantees. 

**PID-IMVF** abandons heuristic gating entirely. We formulate multi-view fusion as a **Partial Information Decomposition (PID)-constrained Nonlinear Independent Component Analysis (ICA)** problem. By deriving a structured factorized encoder directly from the theoretical generative model and employing a differentiable Neural PID Lattice Estimator, our framework guarantees the exact recovery of shared and view-specific latent factors while explicitly quantifying redundancy, uniqueness, and synergy.

---

## 🎯 Issues Resolved & Importance

| Issue | Description & Importance | How PID-IMVF Resolves It |
| :--- | :--- | :--- |
| **Heuristic Decoupling Fallacy** | Prior models (e.g., DMF-Net) assume architectural separation equals information-theoretic separation. This leads to representation contamination, where redundant majority-class features suppress minority-class signals. | Enforces a **strict block-diagonal factorization** $(\hat{S}_v, \hat{U}_v)$ via a Factorized Invertible Neural Network (INN) and Hilbert-Schmidt Independence Criterion (HSIC), guaranteeing mathematically rigorous latent disentanglement. |
| **Transductive Contradiction** | Methods like DMF-Net, HGNN, and ECMGD rely on $\mathcal{O}(N^2)$ global attention or graph adjacency matrices. This is physically intractable for large datasets ($N=30,000$) and violates strict inductive evaluation protocols. | Operates entirely **graph-free** with strict $\mathcal{O}(N \cdot V \cdot d)$ inductive scalability, processing each sample independently without pairwise $N \times N$ interaction matrices. |
| **Class-Imbalance Blindness** | Heuristic fusion gates are highly susceptible to majority-class dominance, causing catastrophic drops in Macro-F1 (e.g., a 15-point gap on the long-tail Animals dataset). | Utilizes a Variational Information Bottleneck (VIB) to filter non-predictive noise and explicitly isolates minority-class discriminative cues into the unique subspace $\hat{U}_v$. |
| **Missing-View Fragility** | Maximizing synergy without bounds makes representations highly vulnerable to test-time view dropout, as synergistic information is irrecoverably lost. | Restricts the downstream classifier exclusively to the partial consensus $\hat{S}^{(\text{obs})}$ and applies a synergy penalty ($\lambda_{\text{syn}}$), strictly bounding performance degradation as proven in Theorem 3. |

---

## ⚙️ Methodology Summary
The PID-IMVF framework operates through a rigorously defined pipeline:
1. **Submanifold Invertible Encoding**: High-dimensional observations $X_v$ are mapped onto a low-dimensional data submanifold $\mathcal{M}_v$ via an INN encoder $g_v$. The projected Jacobian maintains full row rank, ensuring exact change-of-variables without requiring computationally intractable global invertibility in $\mathbb{R}^{d_v}$.
2. **Cross-View Consensus & Identifiability**: Shared factors $\hat{S}_v$ are aligned via a Euclidean Fréchet mean. An HSIC regularizer $\mathcal{L}_{\text{id}}$ enforces strict nonlinear independence between $\hat{S}_v$ and $\hat{U}_v$.
3. **Differentiable Neural PID Estimator**: Instead of unstable adversarial critics (e.g., MINE/NWJ), we employ the Barber-Agakov classifier-based mutual information bound to tractably estimate the Williams-Beer PID atoms (Redundancy, Uniqueness, Synergy) for continuous, high-dimensional variables.
4. **Missing-View Robust Inference**: The classifier $h_\phi$ is strictly restricted to operate on the partial consensus $\hat{S}^{(\text{obs})}$, explicitly excluding view-specific factors $\hat{U}_v$ to guarantee stable predictions under severe observation constraints.

---

## 📊 Experimental Results
Evaluated across 6 diverse benchmarks against 13 state-of-the-art baselines (including DMF-Net, HGNN, TMC, MvIB, and SPICE). All evaluations enforce strict inductive protocols and Bonferroni-corrected statistical significance testing ($\alpha_{\text{adj}} \approx 0.0083$).

| Dataset | Metric | Best Baseline (DMF-Net) | **PID-IMVF (Ours)** | Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Animals** (Long-tail) | Macro-F1 (%) | 71.21 ± 0.71 | **81.35 ± 0.65** | **+10.14%** |
| **100leaves** (Scarce Labels) | Macro-F1 (%) | 86.52 ± 1.05 | **88.45 ± 0.88** | **+1.93%** |
| **HW** (6 Heterogeneous Views) | Macro-F1 (%) | 95.54 ± 0.56 | **95.98 ± 0.48** | **+0.44%** |
| **NoisyMNIST** (Scalability) | Peak GPU Mem | >24.0 GB (OOM) | **8.2 GB** | **Avoids OOM** |
| **HW** (50% Missing Views) | Macro-F1 Drop | -14.09% (Concat) | **-4.56%** | **+9.53% Robustness** |

*Note: PID-IMVF achieves statistically significant gains ($p < 0.0083$) across complex and imbalanced datasets, while maintaining strict $\mathcal{O}(NVD)$ asymptotic complexity.*

---

## 🗄️ Dataset Details
The `data/` module automatically handles downloading and preprocessing. All datasets are used in accordance with their respective licenses and ethical guidelines.

| Dataset | Classes ($C$) | Samples / Views | Feature Dimensions | Valid Reference / Source |
| :--- | :---: | :--- | :--- | :--- |
| **100leaves** | 100 | 1,600 / 3 views | 64, 64, 64 | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/203/one+hundred+plant+species+leaves+data+set) |
| **Animals (AwA2)** | 50 | 10,158 / 2 views | 4096, 4096 | [CVML Group, IST Austria](https://cvml.ist.ac.at/AwA2/) |
| **HW (Handwritten)** | 10 | 2,000 / 6 views | 76, 216, 16, 240, 47, 64 | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/134/multiple+features) |
| **UCI (Multiple Features)** | 10 | 2,000 / 3 views | 240, 76, 6 | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/134/multiple+features) |
| **MNIST** | 10 | 10,000 / 3 views | 784, 784, 784 | [Y. LeCun et al. (1998)](http://yann.lecun.com/exdb/mnist/) |
| **NoisyMNIST** | 10 | 30,000 / 2 views | 784, 784 | [Wang et al., Multi-View Learning Survey](https://github.com/zhaoxin94/awsome-domain-adaptation) |

---

## 🚀 Installation & Quick Start

### 1. Environment Setup
We recommend using Conda to manage dependencies.
```bash
git clone https://github.com/YourUsername/pid-imvf.git
cd pid-imvf
conda env create -f environment.yml
conda activate pid-imvf
```

### 2. Training
Train the model on a specific dataset using the provided configuration files. For example, to train on the Animals dataset:
```bash
python train.py --config configs/datasets/animals.yaml
```
*Hyperparameters ($\lambda_{\text{pred}}$, $\lambda_{\text{noise}}$, $\lambda_{\text{syn}}$) are pre-tuned in the YAML files based on validation Macro-F1.*

### 3. Evaluation & Missing-View Robustness
Evaluate the trained model and simulate missing-view scenarios (e.g., dropping 50% of views at test time):
```bash
python evaluate.py --config configs/datasets/animals.yaml --missing-view --drop-rate 0.5
```

### 4. Reproducing Ablation Studies
Run the complete ablation study pipeline (PID component removal, MLP vs. INN, classifier input restrictions):
```bash
bash scripts/ablation_study.sh
```

---

## 📜 Citation
If you find this work or code useful in your research, please cite our paper:

```bibtex
@article{iqbal2024identifiable,
  title={Identifiable Synergistic Multi-View Learning via Partial Information Decomposition and Nonlinear ICA},
  author={Iqbal, Saeed},
  journal={arXiv preprint},
  year={2024}
}
```

---

## 📬 Contact
For questions, issues, or collaborations, please open an issue in this repository or contact:  
**Saeed Iqbal**  
College of Mechatronics and Control Engineering, Shenzhen University, China  
📧 saeed.iqbal@szu.edu.cn  
🔗 [ORCID: 0000-0002-3176-4658](https://orcid.org/0000-0002-3176-4658)
