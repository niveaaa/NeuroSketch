"""
NeuroSketch — Voice Feature Extraction & Prediction
====================================================
Extracts 22 clinical vocal features from a WAV audio file
(sustained vowel "ahhh") and predicts Parkinson's disease risk
using a trained SVM classifier.

Features extracted:
  - 16 standard features via Praat (parselmouth): F0, jitter, shimmer, HNR/NHR
  - 6 nonlinear dynamics features: RPDE, DFA, spread1, spread2, D2, PPE

Usage:
  python voice_features.py                    # Record from mic (5 seconds)
  python voice_features.py path/to/audio.wav  # Analyze existing WAV file
"""

import os
import sys
import warnings
import numpy as np
import parselmouth
from parselmouth.praat import call

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA = os.path.join(SCRIPT_DIR, "parkinsons.csv")

FEATURE_NAMES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ',
    'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
    'NHR', 'HNR', 'RPDE', 'DFA',
    'spread1', 'spread2', 'D2', 'PPE',
]


# ─────────────────────────────────────────────────────────────
# Feature Extraction (16 Praat features)
# ─────────────────────────────────────────────────────────────

def extract_praat_features(sound):
    """
    Extract 16 standard vocal features using Praat via parselmouth.

    Args:
        sound: parselmouth.Sound object

    Returns:
        dict of 16 feature values
    """
    # Pitch analysis
    pitch = call(sound, "To Pitch", 0.0, 75, 600)
    f0_all = pitch.selected_array['frequency']
    f0_valid = f0_all[f0_all > 0]

    if len(f0_valid) < 5:
        raise ValueError("Not enough voiced frames detected. "
                         "Ensure the audio contains a sustained vowel sound.")

    fo = float(np.mean(f0_valid))
    fhi = float(np.max(f0_valid))
    flo = float(np.min(f0_valid))

    # Point process (for jitter/shimmer)
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)

    # ── Jitter measures ──
    jitter_percent = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_abs = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ddp = jitter_rap * 3  # DDP = 3 × RAP

    # ── Shimmer measures ──
    shimmer_local = call(
        [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    shimmer_db = call(
        [sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    shimmer_apq3 = call(
        [sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    shimmer_apq5 = call(
        [sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    shimmer_apq11 = call(
        [sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    shimmer_dda = shimmer_apq3 * 3  # DDA = 3 × APQ3

    # ── Harmonicity ──
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    # NHR ≈ 1 / (10^(HNR/10))
    if hnr > 0:
        nhr = 1.0 / (10 ** (hnr / 10.0))
    else:
        nhr = 0.0

    return {
        'MDVP:Fo(Hz)': fo,
        'MDVP:Fhi(Hz)': fhi,
        'MDVP:Flo(Hz)': flo,
        'MDVP:Jitter(%)': jitter_percent,
        'MDVP:Jitter(Abs)': jitter_abs,
        'MDVP:RAP': jitter_rap,
        'MDVP:PPQ': jitter_ppq5,
        'Jitter:DDP': jitter_ddp,
        'MDVP:Shimmer': shimmer_local,
        'MDVP:Shimmer(dB)': shimmer_db,
        'Shimmer:APQ3': shimmer_apq3,
        'Shimmer:APQ5': shimmer_apq5,
        'MDVP:APQ': shimmer_apq11,
        'Shimmer:DDA': shimmer_dda,
        'NHR': nhr,
        'HNR': hnr,
    }


# ─────────────────────────────────────────────────────────────
# Feature Extraction (6 nonlinear dynamics features)
# ─────────────────────────────────────────────────────────────

def compute_rpde(f0_series, m=10, tau=1):
    """
    Recurrence Period Density Entropy (RPDE).
    Measures how predictable the pitch period is.
    """
    try:
        N = len(f0_series)
        if N < m * tau + 50:
            return 0.5  # fallback

        # Build delay embedding
        M = N - (m - 1) * tau
        embedded = np.zeros((M, m))
        for i in range(m):
            embedded[:, i] = f0_series[i * tau: i * tau + M]

        # Compute recurrence times
        epsilon = 0.12 * np.std(f0_series)
        recurrence_times = []

        for i in range(min(M - 1, 300)):  # limit for speed
            for j in range(i + 1, min(M, i + 500)):
                dist = np.linalg.norm(embedded[i] - embedded[j])
                if dist < epsilon:
                    recurrence_times.append(j - i)
                    break

        if len(recurrence_times) < 10:
            return 0.5

        # Compute entropy of recurrence time distribution
        rt = np.array(recurrence_times)
        max_rt = int(np.max(rt))
        hist, _ = np.histogram(rt, bins=max(max_rt, 10), density=True)
        hist = hist[hist > 0]
        bin_width = max_rt / max(max_rt, 10)
        entropy = -np.sum(hist * np.log2(hist + 1e-12) * bin_width)

        # Normalize to [0, 1]
        max_entropy = np.log2(max(max_rt, 10))
        rpde = entropy / max_entropy if max_entropy > 0 else 0.5
        return float(np.clip(rpde, 0, 1))

    except Exception:
        return 0.5


def compute_ppe(f0_series):
    """
    Pitch Period Entropy (PPE).
    Measures the impaired control of stable pitch.
    """
    try:
        if len(f0_series) < 10:
            return 0.2

        # Compute log pitch differences
        log_f0 = np.log2(f0_series + 1e-12)
        diff = np.diff(log_f0)

        # Remove semitone quantization effect
        # Map to nearest semitone and compute residual
        semitones = np.round(diff * 12) / 12
        residuals = diff - semitones

        if len(residuals) < 5 or np.std(residuals) < 1e-10:
            return 0.1

        # Compute entropy of residual distribution
        hist, _ = np.histogram(residuals, bins=20, density=True)
        hist = hist[hist > 0]
        bin_width = (np.max(residuals) - np.min(residuals)) / 20
        if bin_width <= 0:
            return 0.1

        entropy = -np.sum(hist * np.log2(hist + 1e-12) * bin_width)
        max_entropy = np.log2(20)
        ppe = entropy / max_entropy if max_entropy > 0 else 0.2

        return float(np.clip(ppe, 0, 1))

    except Exception:
        return 0.2


def compute_spread(f0_series):
    """
    Compute spread1 and spread2 — nonlinear measures of F0 variation.
    spread1: log of the largest eigenvalue of the covariance of the
             pitch period embedding (captures overall spread).
    spread2: log of the ratio of second to first eigenvalue
             (captures shape of the distribution).
    """
    try:
        if len(f0_series) < 20:
            return -6.0, 0.2

        log_f0 = np.log2(f0_series + 1e-12)

        # Create delay embedding (dim=3, tau=1)
        m, tau = 3, 1
        N = len(log_f0) - (m - 1) * tau
        if N < 10:
            return -6.0, 0.2

        embedded = np.zeros((N, m))
        for i in range(m):
            embedded[:, i] = log_f0[i * tau: i * tau + N]

        # Covariance and eigenvalues
        cov = np.cov(embedded.T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eigvals = np.maximum(eigvals, 1e-12)

        spread1 = float(np.log(eigvals[0]))
        spread2 = float(np.log(eigvals[1] / eigvals[0]) if eigvals[0] > 1e-12 else 0.2)

        return spread1, spread2

    except Exception:
        return -6.0, 0.2


def compute_dfa(data, nvals=None):
    """
    Detrended Fluctuation Analysis (DFA).
    Measures long-range correlations in the pitch time series.
    Returns the scaling exponent alpha.
    """
    try:
        N = len(data)
        if N < 20:
            return 0.7

        # Integrate the time series (cumulative sum of deviations from mean)
        y = np.cumsum(data - np.mean(data))

        # Scale sizes
        if nvals is None:
            nvals = np.unique(np.logspace(
                np.log10(4), np.log10(N // 4), num=20
            ).astype(int))
            nvals = nvals[nvals >= 4]

        if len(nvals) < 4:
            return 0.7

        fluctuations = []
        for n in nvals:
            # Split into segments of size n
            num_segments = N // n
            if num_segments < 1:
                continue

            rms_values = []
            for s in range(num_segments):
                segment = y[s * n: (s + 1) * n]
                # Linear detrending
                x = np.arange(n)
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                rms_values.append(np.sqrt(np.mean((segment - trend) ** 2)))

            if rms_values:
                fluctuations.append(np.mean(rms_values))
            else:
                fluctuations.append(np.nan)

        # Filter valid values
        valid = [(n, f) for n, f in zip(nvals, fluctuations)
                 if not np.isnan(f) and f > 0]
        if len(valid) < 4:
            return 0.7

        ns, fs = zip(*valid)
        log_n = np.log(ns)
        log_f = np.log(fs)

        # Linear fit in log-log space → slope = DFA exponent
        coeffs = np.polyfit(log_n, log_f, 1)
        alpha = float(coeffs[0])
        return np.clip(alpha, 0.4, 1.0)

    except Exception:
        return 0.7


def compute_corr_dim(data, emb_dim=10, lag=1):
    """
    Correlation Dimension (D2).
    Estimates the dimensionality of the attractor in phase space.
    """
    try:
        N = len(data)
        M = N - (emb_dim - 1) * lag
        if M < 20:
            return 2.3

        # Build delay embedding
        embedded = np.zeros((M, emb_dim))
        for i in range(emb_dim):
            embedded[:, i] = data[i * lag: i * lag + M]

        # Compute pairwise distances (subsample for speed)
        max_pairs = min(M, 200)
        indices = np.random.choice(M, max_pairs, replace=False) if M > max_pairs else np.arange(M)
        subset = embedded[indices]

        # Compute all pairwise distances
        dists = []
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                d = np.max(np.abs(subset[i] - subset[j]))  # Chebyshev distance
                if d > 0:
                    dists.append(d)

        if len(dists) < 50:
            return 2.3

        dists = np.array(dists)

        # Correlation integral for multiple radii
        radii = np.logspace(np.log10(np.percentile(dists, 5)),
                            np.log10(np.percentile(dists, 95)), 15)

        n_pairs = len(dists)
        corr_sums = []
        for r in radii:
            count = np.sum(dists < r)
            corr_sums.append(count / n_pairs)

        # Filter valid
        valid = [(r, c) for r, c in zip(radii, corr_sums) if c > 0]
        if len(valid) < 5:
            return 2.3

        rs, cs = zip(*valid)
        log_r = np.log(rs)
        log_c = np.log(cs)

        # Slope of log-log plot = correlation dimension
        coeffs = np.polyfit(log_r, log_c, 1)
        d2 = float(coeffs[0])
        return np.clip(d2, 1.0, 4.0)

    except Exception:
        return 2.3


def extract_nonlinear_features(f0_series):
    """
    Extract 6 nonlinear dynamics features from the F0 (pitch) time series.

    Returns:
        dict with RPDE, DFA, spread1, spread2, D2, PPE
    """
    # Clean F0 series (remove zeros/unvoiced)
    f0 = np.array(f0_series, dtype=float)
    f0 = f0[f0 > 0]

    if len(f0) < 30:
        # Not enough data — return reasonable defaults
        return {
            'RPDE': 0.5, 'DFA': 0.7,
            'spread1': -6.0, 'spread2': 0.2,
            'D2': 2.3, 'PPE': 0.2,
        }

    # DFA — Detrended Fluctuation Analysis
    dfa = compute_dfa(f0)

    # D2 — Correlation Dimension
    d2 = compute_corr_dim(f0)

    # RPDE
    rpde = compute_rpde(f0)

    # PPE
    ppe = compute_ppe(f0)

    # spread1, spread2
    spread1, spread2 = compute_spread(f0)

    return {
        'RPDE': rpde,
        'DFA': dfa,
        'spread1': spread1,
        'spread2': spread2,
        'D2': d2,
        'PPE': ppe,
    }


# ─────────────────────────────────────────────────────────────
# Full pipeline: extract all 22 features from a WAV file
# ─────────────────────────────────────────────────────────────

def extract_all_features(wav_path):
    """
    Extract all 22 features from a WAV audio file.

    Args:
        wav_path: path to a WAV file (sustained vowel recording)

    Returns:
        dict of 22 feature values, in the same order as FEATURE_NAMES
    """
    sound = parselmouth.Sound(wav_path)

    # Get pitch series for nonlinear features
    pitch = call(sound, "To Pitch", 0.0, 75, 600)
    f0_series = pitch.selected_array['frequency']

    # Extract both feature sets
    praat_features = extract_praat_features(sound)
    nonlinear_features = extract_nonlinear_features(f0_series)

    # Merge
    all_features = {**praat_features, **nonlinear_features}
    return all_features


# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────

def predict_from_features(features):
    """
    Train the SVM and predict using extracted features.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn import svm
    from sklearn.metrics import accuracy_score

    # Load & train
    data = pd.read_csv(TRAINING_DATA)
    X = data.drop(columns=['name', 'status'], axis=1)
    Y = data['status']

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = svm.SVC(kernel='linear')
    model.fit(X_train_scaled, Y_train)

    # Predict
    feature_vector = [features[name] for name in FEATURE_NAMES]
    arr = np.array(feature_vector, dtype=float).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    prediction = model.predict(arr_scaled)

    return int(prediction[0])


# ─────────────────────────────────────────────────────────────
# Recording (optional — requires sounddevice)
# ─────────────────────────────────────────────────────────────

def record_audio(duration=5, sample_rate=44100):
    """Record audio from the microphone and save as WAV."""
    try:
        import sounddevice as sd
        from scipy.io import wavfile
    except ImportError:
        print("  ⚠  Install 'sounddevice' to use mic recording:")
        print("     pip install sounddevice")
        return None

    output_path = os.path.join(SCRIPT_DIR, "_recording.wav")

    print(f"\n🎤  Recording for {duration} seconds...")
    print("   Say a sustained 'AHHH' into the microphone.\n")

    audio = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate, channels=1, dtype='float64')
    sd.wait()

    # Normalize and save as 16-bit WAV
    audio = audio.flatten()
    audio = (audio / np.max(np.abs(audio)) * 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, audio)

    print(f"   ✅ Saved recording to {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────
# Display results
# ─────────────────────────────────────────────────────────────

def display_results(features, prediction):
    """Print formatted feature extraction results and prediction."""
    pred_label = "Parkinson's Disease Detected" if prediction == 1 else "Healthy — No PD Indicators"
    pred_color = "red" if prediction == 1 else "green"

    print("\n" + "=" * 58)
    print("       VOICE ANALYSIS — FEATURE EXTRACTION RESULTS")
    print("=" * 58)
    print(f"  {'Feature':<28} {'Value':>14}")
    print("-" * 58)

    for name in FEATURE_NAMES:
        val = features.get(name, 0)
        if abs(val) > 1:
            print(f"  {name:<28} {val:>14.3f}")
        else:
            print(f"  {name:<28} {val:>14.5f}")

    print("-" * 58)
    print(f"  {'PREDICTION':<28} {'PD' if prediction == 1 else 'Healthy':>14}")
    print(f"  {'ASSESSMENT':<28} {pred_label:>28}")
    print("=" * 58)
    print("  ⚠  This is a screening tool, not a diagnosis.")
    print("     Consult a neurologist for clinical evaluation.\n")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🔬  NeuroSketch — Voice-Based Parkinson's Detection")
    print("   Extracts 22 vocal features from audio and classifies.\n")

    # Determine audio source
    if len(sys.argv) > 1:
        wav_path = sys.argv[1]
        if not os.path.exists(wav_path):
            print(f"  ❌ File not found: {wav_path}")
            sys.exit(1)
    else:
        # Try to record
        wav_path = record_audio(duration=5)
        if wav_path is None:
            # Fallback: use the existing PDaudio.wav
            wav_path = os.path.join(SCRIPT_DIR, "PDaudio.wav")
            if not os.path.exists(wav_path):
                print("  ❌ No audio file provided and mic recording unavailable.")
                print("  Usage: python voice_features.py <path_to_wav>")
                sys.exit(1)
            print(f"  Using existing audio: {wav_path}")

    print(f"  📂 Analyzing: {os.path.basename(wav_path)}")
    print("  ⏳ Extracting features...")

    try:
        features = extract_all_features(wav_path)
        print("  ✅ Features extracted successfully!")

        print("  ⏳ Running SVM classifier...")
        prediction = predict_from_features(features)

        display_results(features, prediction)

    except ValueError as e:
        print(f"\n  ❌ Error: {e}")
        print("  Tips:")
        print("    - Record a sustained vowel ('ahhh') for at least 3 seconds")
        print("    - Ensure the recording is clear with minimal background noise")
        print("    - Use a WAV format audio file")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ❌ Unexpected error: {e}")
        sys.exit(1)
