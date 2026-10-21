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
├── spam_detector_real.py  # Model training, evaluation, and serialization script
├── predict.py             # Lightweight inference script (runs without dataset!)
├── spam_model.pkl         # Saved MultinomialNB trained model
├── vectorizer.pkl         # Saved CountVectorizer vocabulary
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

### 1. Training the Model
Train the classifier, evaluate accuracy, and save `spam_model.pkl` and `vectorizer.pkl`:
```bash
python spam_detector_real.py
```

### 2. Making Predictions (No Dataset Required!)
Use `predict.py` to classify messages directly:

**Via Command-Line Argument:**
```bash
python predict.py "Congratulations! You have won a $1,000 Walmart Gift Card."
```

**Interactive Terminal Mode:**
```bash
python predict.py
```
```text
Enter SMS: Urgent! Please call our customer service to claim your reward.
-> Result: [SPAM] (99.85% confidence)

Enter SMS: Are we still meeting for lunch today?
-> Result: [HAM] (99.98% confidence)
```

---

## 📚 Dataset Information

The dataset is sourced from the **SMS Spam Collection v.1** corpus, consisting of 5,574 English SMS messages tagged as either `ham` (legitimate) or `spam`. For full details on compilation and sources, refer to the included `readme` file.
