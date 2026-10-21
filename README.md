# SMS Spam & Ham Detector 📩🚫

A fast and accurate Machine Learning text classification model that identifies whether an incoming SMS message is **Ham** (legitimate) or **Spam** (junk / fraudulent).

---

## 🚀 Overview

Spam messages and phishing attempts via SMS are a common nuisance and security risk. This project implements a Natural Language Processing (NLP) pipeline that processes raw text messages, transforms them into numerical vectors, and classifies them using a **Multinomial Naive Bayes** classifier.

### Key Highlights
- **Model:** Multinomial Naive Bayes (`MultinomialNB`)
- **Feature Extraction:** Bag-of-Words (`CountVectorizer`)
- **Dataset:** 5,574 labeled real-world SMS messages
- **Performance:** **~98.7% Accuracy** with balanced precision and recall

---

## 📊 Model Performance

| Metric | Class 0 (Ham) | Class 1 (Spam) | Overall |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.99 | 0.96 | - |
| **Recall** | 0.99 | 0.94 | - |
| **F1-Score** | 0.99 | 0.95 | - |
| **Accuracy** | - | - | **~98.7%** |

---

## 🛠️ Project Structure

```text
├── SMSSpamCollection     # SMS spam research corpus (5,574 tagged messages)
├── readme                 # Original dataset documentation & corpus statistics
├── spam_detector_real.py  # Model training, evaluation, and inference script
├── requirements.txt       # Project dependencies
├── .gitignore             # Ignored temporary files & virtual environments
└── README.md              # Project documentation
```

---

## 💻 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/divyanshshrivastava/spam-ham-detector.git
   cd spam-ham-detector
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Usage

Run the training and evaluation script:
```bash
python spam_detector_real.py
```

### Testing Custom Messages
You can test custom messages inside `spam_detector_real.py`:
```python
sample = ["Congratulations! You’ve been selected for a FREE MacBook Pro. Click here to claim your prize"]
sample_vector = vectorizer.transform(sample)
prediction = model.predict(sample_vector)
print("Result:", "Spam" if prediction[0] == 1 else "Ham")
```

---

## 📚 Dataset Information

The dataset is sourced from the **SMS Spam Collection v.1** corpus, consisting of 5,574 English SMS messages tagged as either `ham` (legitimate) or `spam`. For full details on compilation and sources, refer to the included `readme` file.
