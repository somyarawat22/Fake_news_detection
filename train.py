import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from utils.preprocess import load_data
from utils.tokenizer_utils import tokenize_data

# Load data
data = load_data()

X = data['text']
y = data['label']

# 🔥 CHECK DATA BALANCE (optional but useful)
print("Label count:\n", data['label'].value_counts())

# Tokenize
X = tokenize_data(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
