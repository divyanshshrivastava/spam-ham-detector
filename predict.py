import os
import sys
import joblib

MODEL_PATH = "spam_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print(f"Error: Model files ('{MODEL_PATH}' or '{VECTORIZER_PATH}') not found.")
        print("Please run 'python spam_detector_real.py' once to train and save the model.")
        sys.exit(1)
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def predict_message(text: str, model, vectorizer):
    text_vector = vectorizer.transform([text])
    pred = model.predict(text_vector)[0]
    proba = model.predict_proba(text_vector)[0]
    
    label = "SPAM" if pred == 1 else "HAM"
    confidence = proba[pred] * 100
    return label, confidence

def main():
    model, vectorizer = load_artifacts()

    # If message passed as CLI argument: python predict.py "Free cash win now"
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        label, conf = predict_message(message, model, vectorizer)
        print(f"\nMessage: \"{message}\"")
        print(f"Prediction: [{label}] (Confidence: {conf:.2f}%)\n")
        return

    # Interactive mode
    print("=" * 50)
    print("      SMS SPAM / HAM DETECTOR (Inference Mode)")
    print("      (Runs without dataset using saved model)")
    print("=" * 50)
    print("Type a message to classify (or 'exit' / 'q' to quit):\n")

    while True:
        try:
            msg = input("Enter SMS: ").strip()
            if not msg:
                continue
            if msg.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            
            label, conf = predict_message(msg, model, vectorizer)
            print(f"-> Result: [{label}] ({conf:.2f}% confidence)\n")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
