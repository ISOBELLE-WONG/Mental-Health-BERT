# Mental Health BERT 🧠

A fine-tuned **BERT-based multi-class classification model** for detecting mental health signals from text, with a strong focus on **model evaluation and interpretability**.

This project demonstrates how transformer-based language models can be applied to mental health signal recognition while maintaining transparency through confidence analysis, ROC-AUC metrics, and local interpretability techniques.


## 🔍 Project Overview

Mental health–related language often contains nuanced, ambiguous, and emotionally rich signals.  
This project applies a **fine-tuned BERT model** to classify mental health signals into multiple categories and evaluates the model using robust, interpretable metrics.

Key goals:
- Accurate **multi-class classification** of mental health signals
- Careful **confidence and error analysis**
- Emphasis on **interpretability** rather than black-box prediction


## 🧠 Model & Methodology

- **Model**: Fine-tuned BERT (Hugging Face Transformers)
- **Task**: Multi-class text classification
- **Evaluation**:
  - One-vs-Rest ROC–AUC (macro & per-class)
  - Confidence-based error analysis
- **Interpretability**:
  - Local explanation techniques (e.g., LIME-style analysis)
  - Examination of incorrect predictions vs confidence levels

The evaluation highlights that most incorrect predictions occur at **moderate confidence levels**, suggesting the model appropriately expresses uncertainty rather than failing silently.


## 📂 Repository Structure

