from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


def build_lstm_model(input_shape: tuple[int, int], units: int = 64) -> Any:
    """Build and compile LSTM model."""
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.30))
    model.add(LSTM(units=32))
    model.add(Dropout(0.30))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_lstm_model(
    X_train: NDArray[Any],
    y_train: NDArray[Any],
    X_test: NDArray[Any],
    y_test: NDArray[Any],
    epochs: int = 100,
    batch_size: int = 32,
    prediction_threshold: float = 0.55,
) -> tuple[Any, Any, NDArray[Any], NDArray[Any]]:
    """Train LSTM model and evaluate."""
    # Class weights for imbalanced data
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=1,
    )

    # Predictions
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob >= prediction_threshold).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"LSTM Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return model, history, y_pred_prob, y_pred
