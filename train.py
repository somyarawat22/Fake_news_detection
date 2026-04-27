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

# 🔥 HANDLE IMBALANCE
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# Model
model = Sequential([
    Embedding(5000, 64),
    LSTM(64),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 🔥 TRAIN
history = model.fit(
    X_train,
    y_train,
    epochs=8,
    validation_data=(X_test, y_test),
    class_weight=class_weights
)

# Evaluate
y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Train', 'Validation'])
plt.title("Accuracy Graph")
plt.show()

# Save
model.save('model/lstm_model.h5')
