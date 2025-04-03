import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import binarize
from sklearn.metrics import accuracy_score

class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate=0.1):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.weights = np.random.normal(0, 0.1, (n_visible, n_hidden))
        self.h_bias = np.zeros(n_hidden)
        self.v_bias = np.zeros(n_visible)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, data, epochs=10, batch_size=10):
        for _ in range(epochs):
            np.random.shuffle(data)
            for batch_start in range(0, len(data), batch_size):
                batch = data[batch_start:batch_start + batch_size]
                pos_hidden_probs = self.sigmoid(batch @ self.weights + self.h_bias)
                pos_associations = batch.T @ pos_hidden_probs
                neg_visible_probs = self.sigmoid(pos_hidden_probs @ self.weights.T + self.v_bias)
                neg_hidden_probs = self.sigmoid(neg_visible_probs @ self.weights + self.h_bias)
                neg_associations = neg_visible_probs.T @ neg_hidden_probs
                
                self.weights += self.lr * (pos_associations - neg_associations) / len(batch)
                self.v_bias += self.lr * np.mean(batch - neg_visible_probs, axis=0)
                self.h_bias += self.lr * np.mean(pos_hidden_probs - neg_hidden_probs, axis=0)

    def extract_features(self, data):
        return self.sigmoid(data @ self.weights + self.h_bias)

# Load and prepare data
digits = load_digits()
X = digits.data / 16.0  # Normalize
X = binarize(X, threshold=0.5)
y = digits.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RBM
rbm = RBM(n_visible=64, n_hidden=50)
rbm.train(X_train, epochs=20, batch_size=10)

# Extract features
train_features = rbm.extract_features(X_train)
test_features = rbm.extract_features(X_test)

# Train classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(train_features, y_train)

# Evaluate
y_pred = classifier.predict(test_features)
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification accuracy: {accuracy:.4f}")

# Print shapes for verification
print(f"Original data shape: {X_train.shape}")
print(f"Extracted features shape: {train_features.shape}")