# 🧠 MemoryART: An LLM Healthcare Assistant Memory Framework Based on Adaptive Resonance Theory

## 📖 Project Overview

**MemoryART** is a cognitively inspired framework for enhancing both short-term and long-term memory in large language models (LLMs). To address bottlenecks that LLMs face in multi-turn medical consultations—such as **prototype collapse** —we introduce **Adaptive Resonance Theory (ART)**.

This project is more than a simple RAG system. By simulating human-like memory processing, it transforms unstructured doctor–patient dialogues into structured, scenario-based memories with **temporal awareness**, **motivation awareness**, and **event-level features**.

---

## 🏛️ Core Architecture: A Triadic Memory Storage System

Inspired by classic cognitive science models, MemoryART organizes LLM memory into three collaborative modules:

### 1. Working Memory (WM)

- **Purpose**: Handles the current consultation request.
- **Implementation**: Extracts “resonance cues” from the current query (e.g., patient identity, current symptoms, recent dates) as triggers for retrieval.

### 2. Episodic Memory (EM)

- **Purpose**: Stores the patient’s historical experiences.
- **Innovation**: Translates raw conversation logs into **Structured Events** by **Adaptive Resonance Theory (ART)**.
- **Representation**: JSON-style event records with fields such as `Time`, `Diagnosis`, `Symptoms`, `Treatment`, and `Motivation`.

### 3. Semantic Memory (SM)

- **Purpose**: Maintains long-horizon patient profiles (User Profiles).
- **Implementation**: Aggregates static facts extracted across multiple sessions (e.g., allergy history, occupation, lifestyle habits) and supports reasoning via LLM commonsense.

---

## ⚡ Core Mechanism: Adaptive Resonance Retrieval

Unlike traditional single-channel embedding-based retrieval, MemoryART adopts a **multi-channel resonance matching mechanism**:

- **Resonance score computation**: The retriever not only measures semantic similarity, but also computes overlaps across the following dimensions via `resonance_based_retriever.py`:
  - **Identity channel**: Ensures memories belong to the correct subject (prevents identity confusion).
  - **Temporal/Spatial channel**: Filters irrelevant history using temporal logic.
  - **Motivation channel**: Detects the underlying intent behind a patient’s consultation.
- **Prototype preservation**: ART ensures that incorporating new memories does not overwrite or blur salient existing memory features, effectively preventing “drift” of critical medical history during consultations.

---

## 🛠️ Implementation Highlights

- **Multi-model integration**: The framework natively supports **DeepSeek-V3**, **GPT-4**, **ERNIE Bot 4.5**, and **Qwen 235B**, validating MemoryART’s generalization across different foundation models.
- **Memory encoding pipeline**:
  1. **Parse**: Use an LLM to extract key actions from raw dialogues.
  2. **Structuring**: Convert them into standardized event structures.
  3. **Indexing**: Build a semantic index via LlamaIndex, augmented with a hard-constraint rule library.

---

## 📈 Evaluation Metrics

This project follows high-pressure evaluation standards tailored for healthcare consultation scenarios:

- **F1 & BLEU scores**: Measure alignment between generated answers and expert reference answers.
- **Multi-hop QA tests**: Verify whether the model can retrieve and integrate information across multiple historical sessions (e.g., computing the duration since the first asthma diagnosis).
- **MCQ accuracy**: Evaluate the model’s precision in judging the patient’s current pathological state.

## Also follows evaluation on public dataset: Locomo
---

## 🔬 Citation

If you use this project or draw inspiration from the MemoryART paper in your research, please cite:

```bibtex
@inproceedings{memoryart2026,
  title={MemoryART: Enhancing LLMs via Multi-Memory Models with Adaptive Resonance Theory for Healthcare Agents},
  author={Anonymous},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
