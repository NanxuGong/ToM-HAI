<div align="center">

# 🧠 ToM-HAI: Theory of Mind on Human-AI Symbiosis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


*From Assistants to Companions: Towards the Usefulness of Improving Theory of Mind for Human-AI Symbiosis*

</div>

---



## 📁 Repository Structure

| File | Description | Type |
|------|-------------|------|
| `exploretom_first.py` | 🔧 SFT training dataset construction | Data Processing |
| `hitom_first.json` | 📊 Raw first-perspective HiToM data [1]| Raw Data |
| `training_exploretom.py` | 🚀 Step 1 of RL training dataset construction | Data Processing |
| `training_hitom.py` | 🎯 Step 2 of RL training dataset construction | Data Processing |

## 🚀 Quick Start

### 📈 Training Data Generation

#### For Reinforcement Learning (RL)
```bash
# Step 1: Generate ExploreToM training data
python training_exploretom.py

# Step 2: Generate HiToM training data  
python training_hitom.py
```

#### For Supervised Fine-Tuning (SFT)
```bash
# Generate SFT training dataset
python exploretom_first.py
```

## 🔬 Training Methodology

| Method | Framework | Description |
|--------|-----------|-------------|
| **SFT** | ExploreToM [2] | Supervised fine-tuning approach |
| **RL** | ToM-RL [3] | Reinforcement learning framework |


## 📚 References

**[1]** He, Yinghui, et al. *"Hi-tom: A benchmark for evaluating higher-order theory of mind reasoning in large language models."* **arXiv preprint arXiv:2310.16755** (2023).

**[2]** Sclar, Melanie, et al. *"Explore theory of mind: Program-guided adversarial data generation for theory of mind reasoning."* **arXiv preprint arXiv:2412.12175** (2024). 

**[3]** Lu, Yi-Long, et al. *"Do Theory of Mind Benchmarks Need Explicit Human-like Reasoning in Language Models?"* **arXiv preprint arXiv:2504.01698** (2025).

---

