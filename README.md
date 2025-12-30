# Transformer Research

**Objective:** Investigate how transformer-based AI models learn to represent and generalize causal structures.

## Architecture
- 2-Layer Transformer (No MLP)
- Continuous Input (No Embeddings)
- Task: Predict next 50 time steps given 100.

## Setup
1. `pip install -r requirements.txt`
2. Run data generation: `python src/data_gen.py`