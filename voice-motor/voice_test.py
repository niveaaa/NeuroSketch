"""
NeuroSketch — Voice-Based Parkinson's Detection
================================================
Trains an SVM classifier on the Oxford Parkinson's Disease dataset
(parkinsons.csv) using 22 vocal features, then evaluates on held-out
test data and provides an interactive prediction mode.

Features used (from sustained phonation recordings):
  - Fundamental frequency measures (Fo, Fhi, Flo)
  - Jitter variants (%, Abs, RAP, PPQ, DDP)
  - Shimmer variants (dB, APQ3, APQ5, APQ, DDA)
  - Noise-to-harmonics ratio (NHR, HNR)
  - Nonlinear dynamics (RPDE, DFA, spread1, spread2, D2, PPE)

Usage:
  python voice_test.py                  # Train, evaluate, and enter interactive mode
  python voice_test.py --test-only      # Only run predictions on 'test data.csv'
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA = os.path.join(SCRIPT_DIR, "parkinsons.csv")
TEST_DATA = os.path.join(SCRIPT_DIR, "test data.csv")

# The 22 feature columns used for prediction (excludes 'name' and 'status')
FEATURE_COLUMNS = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ',
    'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
    'NHR', 'HNR', 'RPDE', 'DFA',
    'spread1', 'spread2', 'D2', 'PPE',
]


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def load_and_train(csv_path, random_state=2, test_size=0.2):
    """
    Load the Oxford Parkinson's dataset and train a linear SVM.

    Returns: (model, scaler, train_accuracy, test_accuracy)
    """
    print("\n🔬  VOICE TEST — Parkinson's Detection via Vocal Features")
    print("=" * 58)

    # ── Load data ──
    data = pd.read_csv(csv_path)
    print(f"  Loaded {len(data)} samples from {os.path.basename(csv_path)}")
    print(f"  PD samples: {(data['status'] == 1).sum()}  |  "
          f"Healthy samples: {(data['status'] == 0).sum()}")

    # ── Feature / target split ──
    X = data.drop(columns=['name', 'status'], axis=1)
    Y = data['status']

    # ── Train / test split ──
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    print(f"  Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

    # ── Standardize ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Train SVM ──
    model = svm.SVC(kernel='linear')
    model.fit(X_train_scaled, Y_train)

    # ── Evaluate ──
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    train_acc = accuracy_score(Y_train, train_pred)
    test_acc = accuracy_score(Y_test, test_pred)

    print("-" * 58)
    print(f"  Train Accuracy: {train_acc * 100:.1f}%")
    print(f"  Test Accuracy:  {test_acc * 100:.1f}%")
    print("-" * 58)

    # Detailed classification report
    print("\n  Classification Report (Test Set):")
    report = classification_report(
        Y_test, test_pred,
        target_names=["Healthy (0)", "Parkinson's (1)"],
        zero_division=0
    )
    for line in report.split('\n'):
        print(f"    {line}")

    print("=" * 58)

    return model, scaler, train_acc, test_acc


# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────

def predict_single(model, scaler, features):
    """
    Predict PD status for a single sample.

    Args:
        features: tuple/list of 22 numeric feature values

    Returns:
        (prediction, label_str)
        prediction: 0 (Healthy) or 1 (Parkinson's)
    """
    arr = np.asarray(features, dtype=float).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled)
    label = "Parkinson's Disease Detected" if pred[0] == 1 else "Healthy — No PD Indicators"
    return int(pred[0]), label


def run_test_data(model, scaler, test_csv_path):
    """
    Run predictions on the external test data CSV.
    The CSV has no header, columns = 22 features + status (last col).
    """
    if not os.path.exists(test_csv_path):
        print(f"\n  ⚠  Test file not found: {test_csv_path}")
        return

    # Load test data — no header, same column layout as parkinsons.csv minus 'name'
    # Column order: 16 features, status, 6 more features (status at index 16)
    all_cols_no_name = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
        'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ',
        'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
        'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
        'NHR', 'HNR',
        'status',
        'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE',
    ]
    test_df = pd.read_csv(test_csv_path, header=None, names=all_cols_no_name)

    X_ext = test_df[FEATURE_COLUMNS]
    Y_ext = test_df['status']

    X_ext_scaled = scaler.transform(X_ext)
    predictions = model.predict(X_ext_scaled)

    acc = accuracy_score(Y_ext, predictions)

    print(f"\n📋  External Test Data: {os.path.basename(test_csv_path)}")
    print("=" * 58)
    print(f"  Samples: {len(test_df)}")
    print(f"  Accuracy: {acc * 100:.1f}%")
    print("-" * 58)

    # Show per-sample results
    correct = 0
    for i, (pred, actual) in enumerate(zip(predictions, Y_ext)):
        status = "✅" if pred == actual else "❌"
        pred_label = "PD" if pred == 1 else "Healthy"
        actual_label = "PD" if actual == 1 else "Healthy"
        if pred == actual:
            correct += 1
        print(f"  Sample {i+1:3d}: Predicted={pred_label:7s}  "
              f"Actual={actual_label:7s}  {status}")

    print("-" * 58)
    print(f"  {correct}/{len(test_df)} correct ({acc * 100:.1f}%)")
    print("=" * 58)


def interactive_mode(model, scaler):
    """
    Let the user input 22 feature values for a prediction.
    """
    print("\n🎤  Interactive Prediction Mode")
    print("=" * 58)
    print("  Enter 22 comma-separated vocal feature values:")
    print(f"  Features: {', '.join(FEATURE_COLUMNS[:6])} ...")
    print("  (Type 'q' to quit)\n")

    while True:
        try:
            user_input = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if user_input.lower() in ('q', 'quit', 'exit'):
            print("  Goodbye!")
            break

        try:
            values = [float(x.strip()) for x in user_input.split(',')]
            if len(values) != 22:
                print(f"  ⚠  Expected 22 values, got {len(values)}. Try again.")
                continue

            pred, label = predict_single(model, scaler, values)
            print(f"\n  {'=' * 40}")
            print(f"  PREDICTION: {pred}")
            print(f"  RESULT:     {label}")
            print(f"  {'=' * 40}\n")

        except ValueError:
            print("  ⚠  Invalid input. Enter 22 comma-separated numbers.")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(TRAINING_DATA):
        print(f"❌ Training data not found: {TRAINING_DATA}")
        sys.exit(1)

    # Train the model
    model, scaler, train_acc, test_acc = load_and_train(TRAINING_DATA)

    # Run on external test data if available
    if os.path.exists(TEST_DATA):
        run_test_data(model, scaler, TEST_DATA)

    # Enter interactive mode unless --test-only flag
    if "--test-only" not in sys.argv:
        interactive_mode(model, scaler)
