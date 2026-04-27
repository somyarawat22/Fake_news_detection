import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = None
tokenizer = None

def load_resources():
    global model, tokenizer

    if model is None:
        print("Loading model...")
        model = load_model('model/lstm_model.h5')
        print("Model loaded!")

    if tokenizer is None:
        with open('model/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)

def predict_news(text):
    load_resources()

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=200)

    pred = model.predict(padded)[0][0]

    print("Prediction value:", pred)

    # 🔥 FIXED THRESHOLD
    if pred > 0.6:
        return f"Real News ({pred*100:.2f}%)"
    else:
        return f"Fake News ({(1-pred)*100:.2f}%)"
