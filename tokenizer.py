from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def tokenize_data(texts, max_len=200):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len)

    # Save tokenizer
    with open('model/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    return padded
