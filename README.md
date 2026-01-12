# Heterogeneous Graph Reasoning with GNNs and LLMs

## 📌 Overview
This repository contains the **codebase of an R&D project** exploring the integration of
**Graph Neural Networks (GNNs)** and **Large Language Models (LLMs)** for
**reasoning, link prediction, and explanation generation on heterogeneous graphs**.

The project aims to bridge:
- **Structured learning** (heterogeneous graph representation with GNNs)
- **Symbolic and semantic reasoning** (path-based reasoning and Chain-of-Thought)
- **Natural language explanation** (LLM-based generation)

This repository includes **code only** (no datasets, no trained models, no LaTeX report).

---

## 🎯 Research Objectives
- Learn expressive node and edge representations from **heterogeneous graphs**
- Perform **link prediction** and **top-k reasoning tasks**
- Extract **reasoning paths** from graphs
- Generate **faithful natural language explanations** using LLMs
- Explore hybrid **GNN + LLM architectures** inspired by recent research

---

## 🧠 High-Level Architecture

Heterogeneous Graph
│
▼
Heterogeneous GNN Encoder
(R-GCN / HGT / HAN)
│
▼
Graph Embeddings
│
├── Link Prediction / Scoring
│
├── Subgraph & Path Extraction
│
▼
LLM-based Explanation Module
(Ollama / LLM prompting)
│
▼
Natural Language Reasoning & Explanation

---

## 📁 Project Structure

hetero-graph-reasoning-gnn-llm/
│
├── data_processing/ # Graph loading, filtering, preprocessing
│ ├── load_hetinet.py
│ └── filter_graph.py
│
├── models/
│ ├── gnn/ # Heterogeneous GNN encoders
│ └── link_prediction/ # Link prediction models
│
├── training/ # Training pipeline
│ ├── prepare_ctd.py
│ └── train_ctd.py
│
├── prediction/ # Inference & top-k prediction
│ └── predict_topk.py
│
├── explanation/ # Path extraction and reasoning logic
│ ├── extract_paths.py
│ └── map_and_print_paths.py
│
├── LLM/ # LLM-based explanation generation
│ └── generate_explanation_ollama.py
│
├── evaluation/ # Evaluation scripts and metrics
│
├── train.py # Main training entry point
├── test_hetinet.py # Testing script
├── README.md
└── .gitignore

---

## ⚙️ Installation

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt

🚀 Training
python train.py

🔍 Prediction
python prediction/predict_topk.py

🧾 Explanation Generation (LLM)

This module generates natural language explanations from extracted graph paths.

python LLM/generate_explanation_ollama.py

🧪 Evaluation
python test_hetinet.py

```

📚 Research Context & Inspiration

This project is inspired by recent advances in LLMs for Graph Learning, including:

GraphLLM – Boosting Graph Reasoning with Large Language Models (ICLR 2024)

HiGPT – Heterogeneous Graph Instruction Tuning for LLMs

HetGCoT – Heterogeneous Graph Chain-of-Thought Reasoning

Large Language Models for Graph Learning: A Survey (KDD 2024)

The code explores hybrid paradigms such as:

GNN-to-LLM alignment

Subgraph-based reasoning

Graph-guided Chain-of-Thought

Faithful explanation generation

⚠️ Notes

Datasets are intentionally excluded.

Trained models (.pt) are not versioned.

The repository focuses on reproducible research code.

👤 Authors

Imane Enneya

Ayoub Fakraoui

📄 License

This project is provided for research and educational purposes.


