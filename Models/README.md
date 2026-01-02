# Mental Health Text Classifier (Fine-Tuned MentalBERT)

A transformer-based model fine-tuned on MentalBERT for four-class mental health text classification. This model identifies **Anxiety**, **Depression**, **Normal**, and **Suicidal** categories from short text inputs. It is designed strictly for **research and academic purposes**, enabling analysis of mental-health-related language in social media and similar text sources.

---

## 📌 Overview
This model adapts **mental/mental-bert-base-uncased**, a domain-specific BERT variant pre-trained on mental health conversations from Reddit. Fine-tuning is performed using a curated and balanced dataset assembled from public Kaggle datasets and Reddit mental health posts.

**Key Capabilities:**
- Multi-class mental health classification
- Robust performance on noisy, user-generated text
- Domain-adapted contextual understanding

**Not intended for clinical diagnosis or real-world medical decision-making.**

---

## 🧠 Model Details
- **Architecture:** BertForSequenceClassification
- **Parameters:** ~110M
- **Base Model:** mental/mental-bert-base-uncased
- **Labels:** Anxiety, Depression, Normal, Suicidal
- **Framework:** PyTorch + Hugging Face Transformers

---

## 📊 Performance
The model achieves high accuracy and stable generalization across mental health classes.

| Metric | Score |
|--------|--------|
| **Accuracy** | 89.72% |
| **Macro F1** | 89.54% |
| **Macro Precision** | 89.56% |
| **Macro Recall** | 89.72% |

**Class-wise F1 Scores:**
- Normal: **0.96**
- Suicidal: **0.94**
- Anxiety: **0.87**
- Depression: **0.82**

---

## 🧪 Example Usage
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "/kaggle/input/mental-health-text-classifier"  # update according to your Kaggle environment

tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

label_map = {0:"Anxiety", 1:"Depression", 2:"Normal", 3:"Suicidal"}

text = "I feel empty and tired all the time."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    logits = model(**inputs).logits
    prediction = logits.argmax().item()

print(label_map[prediction])
```

---

## 🧹 Dataset Summary
This model is fine-tuned on a carefully curated and cleaned dataset composed of:
- Kaggle Suicide Watch dataset
- Kaggle Sentiment Analysis for Mental Health dataset
- Public Reddit mental health posts
- Custom-preprocessed samples

The training data is balanced across the four target classes with normalized labels and text cleaning applied.

Refer to the **Provenance** tab for complete details.

---

## ⚙️ Training Configuration
- **Epochs:** 5
- **Batch Size:** 16 (training), 32 (validation)
- **Learning Rate:** 2e-5
- **Optimizer:** AdamW
- **Max Sequence Length:** 128 tokens
- **Loss Function:** Weighted cross-entropy
- **Scheduler:** Linear warmup
- **Hardware:** Google Colab (Tesla T4 GPU)

---

## 🛑 Limitations & Warnings
- This model is **not** a clinical tool.
- Predictions may inherit biases from Reddit-based training data.
- Not suitable for crisis assessment or emergency intervention.
- Text may be misinterpreted due to ambiguity or sarcasm.

---

## 🔐 Ethical & Responsible Use
- All training data is public and anonymized.
- Use responsibly in research settings.
- Avoid real-world medical or psychological decision-making based on model predictions.

---

## 👤 Author
**Priyangshu Mukherjee**  
BTech CSE, RV University  
Email: priyangshumukherjeebtech24@rvu.edu.in

---

## 📎 Citation
```bibtex
@software{mental_health_classifier_2025,
  author = {Mukherjee, Priyangshu},
  title = {Mental Health Text Classifier (MentalBERT Fine-tuned)},
  year = {2025},
  note = {Fine-tuned model for multi-class mental health text classification}
}
```

---

## 📄 License
The model is released under the **Apache 2.0 License**. The base model and source datasets retain their respective licenses.

Use is limited to **research, academic, and non-commercial domains**.

