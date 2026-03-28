import cv2
import mediapipe as mp
import math
import time
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────

def smooth_signal(signal, window=5):
    """Apply a simple moving average to reduce noise."""
    if len(signal) < window:
        return np.array(signal)
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode='same')


def extract_features(times, distances):
    """
    Extract 6 clinically-relevant features from the finger-tapping
    distance-time signal.

    Returns a dict of features, plus arrays useful for plotting
    (smoothed signal, peak/trough indices, trend line).
    """
    t = np.array(times)
    d = np.array(distances, dtype=float)

    # ── 1. Clean invalid frames (distance == -1) ──
    valid = d > 0
    t_valid = t[valid]
    d_valid = d[valid]

    if len(d_valid) < 10:
        return None  # not enough data

    # ── 2. Smooth the signal ──
    d_smooth = smooth_signal(d_valid, window=7)

    # ── 3. Find peaks (finger open) and troughs (finger closed) ──
    # Adaptive prominence: use 10% of the signal range
    signal_range = np.max(d_smooth) - np.min(d_smooth)
    prominence = max(signal_range * 0.10, 3)

    peak_idx, _ = find_peaks(d_smooth, prominence=prominence, distance=4)
    trough_idx, _ = find_peaks(-d_smooth, prominence=prominence, distance=4)

    if len(peak_idx) < 3:
        return None  # couldn't detect enough taps

    # ── 4. Compute amplitudes (each peak minus its nearest trough) ──
    amplitudes = []
    for pi in peak_idx:
        # find closest trough before this peak
        before = trough_idx[trough_idx < pi]
        if len(before) > 0:
            amp = d_smooth[pi] - d_smooth[before[-1]]
            amplitudes.append(amp)
        else:
            amplitudes.append(d_smooth[pi] - np.min(d_smooth))
    amplitudes = np.array(amplitudes)

    # ── 5. Compute the 6 features ──

    # 5a. Normalized Mean Amplitude (% of signal range)
    #     This removes dependence on camera distance
    mean_amplitude_px = float(np.mean(amplitudes))
    norm_amplitude = float(mean_amplitude_px / signal_range * 100) if signal_range > 0 else 0

    # 5b. Amplitude Decrement — % change from first-half to second-half peaks
    #     More robust than raw slope for noisy webcam data
    peak_times = t_valid[peak_idx]
    peak_values = d_smooth[peak_idx]
    mid = len(peak_values) // 2
    first_half_avg = float(np.mean(peak_values[:mid])) if mid > 0 else 0
    second_half_avg = float(np.mean(peak_values[mid:])) if mid > 0 else 0
    if first_half_avg > 0:
        amplitude_decrement_pct = float(
            (second_half_avg - first_half_avg) / first_half_avg * 100
        )
    else:
        amplitude_decrement_pct = 0.0

    # Trend line for plotting
    if len(peak_times) >= 2:
        slope, intercept = np.polyfit(peak_times, peak_values, 1)
        trend_line = slope * peak_times + intercept
    else:
        trend_line = peak_values

    # 5c. Tap Frequency (Hz)
    total_duration = t_valid[-1] - t_valid[0]
    num_taps = len(peak_idx)
    tap_frequency = float(num_taps / total_duration) if total_duration > 0 else 0.0

    # 5d. Inter-tap intervals — filter outliers with IQR before metrics
    raw_intervals = np.diff(peak_times)
    if len(raw_intervals) > 3:
        q1, q3 = np.percentile(raw_intervals, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_intervals = raw_intervals[
            (raw_intervals >= lower_bound) & (raw_intervals <= upper_bound)
        ]
        if len(filtered_intervals) < 2:
            filtered_intervals = raw_intervals  # fallback
    else:
        filtered_intervals = raw_intervals

    # Mean Velocity (pixels/sec)
    if len(filtered_intervals) > 0 and len(amplitudes) > 1:
        min_len = min(len(amplitudes) - 1, len(filtered_intervals))
        velocities = amplitudes[1:min_len+1] / filtered_intervals[:min_len]
        mean_velocity = float(np.mean(velocities))
    else:
        mean_velocity = 0.0

    # 5e. Rhythm Variability — CoV of FILTERED inter-tap intervals
    if len(filtered_intervals) > 1 and np.mean(filtered_intervals) > 0:
        rhythm_variability = float(
            np.std(filtered_intervals) / np.mean(filtered_intervals)
        )
    else:
        rhythm_variability = 0.0

    # 5f. Hesitation Count — RAW intervals > 2.5× the median of filtered
    if len(raw_intervals) > 1 and len(filtered_intervals) > 0:
        median_interval = np.median(filtered_intervals)
        hesitation_count = int(np.sum(raw_intervals > 2.5 * median_interval))
    else:
        hesitation_count = 0

    features = {
        'mean_amplitude_px': mean_amplitude_px,
        'norm_amplitude': norm_amplitude,
        'amplitude_decrement_pct': amplitude_decrement_pct,
        'tap_frequency': tap_frequency,
        'mean_velocity': mean_velocity,
        'rhythm_variability': rhythm_variability,
        'hesitation_count': hesitation_count,
        'num_taps': num_taps,
    }

    plot_data = {
        't_valid': t_valid,
        'd_smooth': d_smooth,
        'peak_idx': peak_idx,
        'trough_idx': trough_idx,
        'peak_times': peak_times,
        'trend_line': trend_line,
    }

    return features, plot_data


# ─────────────────────────────────────────────────────────────
# Risk Score Computation
# ─────────────────────────────────────────────────────────────

def compute_risk_score(features):
    """
    Compute a heuristic Parkinson's risk score (0-100) from extracted
    features. Higher score = higher risk indication.

    Thresholds are calibrated for WEBCAM + MediaPipe tracking, which
    has lower frame rates and noisier data than clinical sensors.
    A healthy person tapping normally should score 0-20.
    """
    score = 0.0

    # 1. Normalized Amplitude — lower % of range = smaller taps (weight: 20)
    #    Healthy webcam: typically 40-80% of range
    #    PD: taps become very small, < 25% of range
    amp = features['norm_amplitude']
    if amp < 15:
        score += 20
    elif amp < 25:
        score += 12
    elif amp < 35:
        score += 5
    # else: 0 (healthy tap size)

    # 2. Amplitude Decrement — % decay first→second half (weight: 25)
    #    This is the STRONGEST PD indicator (progressive fatigue)
    #    Healthy: within ±15%, PD: often > 30% decay
    dec = features['amplitude_decrement_pct']
    if dec < -40:
        score += 25
    elif dec < -25:
        score += 18
    elif dec < -15:
        score += 8
    # else: no significant decay (normal)

    # 3. Tap Frequency — lower is worse (weight: 20)
    #    Webcam detects fewer peaks than actual taps due to frame rate
    #    Healthy on webcam: typically > 1.0 Hz detected
    #    PD: often < 0.7 Hz detected
    freq = features['tap_frequency']
    if freq < 0.5:
        score += 20
    elif freq < 0.8:
        score += 12
    elif freq < 1.0:
        score += 5
    # else: 0 (fine for webcam)

    # 4. Rhythm Variability — CoV of inter-tap intervals (weight: 20)
    #    After IQR filtering, healthy webcam CoV is typically < 0.4
    #    PD: high irregularity, CoV > 0.6
    rv = features['rhythm_variability']
    if rv > 0.8:
        score += 20
    elif rv > 0.6:
        score += 12
    elif rv > 0.45:
        score += 5
    # else: 0 (acceptable rhythm)

    # 5. Hesitation Count — freezes/pauses (weight: 10)
    #    Using 2.5× median threshold after filtering
    hc = features['hesitation_count']
    if hc >= 5:
        score += 10
    elif hc >= 3:
        score += 6
    elif hc >= 2:
        score += 3
    # else: 0 (normal)

    # 6. Mean Velocity — lower is worse (weight: 5)
    vel = features['mean_velocity']
    if vel < 30:
        score += 5
    elif vel < 60:
        score += 2

    return min(score, 100)


def get_risk_label(score):
    """Return a human-readable risk interpretation."""
    if score <= 15:
        return "Low Risk", "green"
    elif score <= 35:
        return "Mild Indicators", "gold"
    elif score <= 60:
        return "Moderate Risk", "orange"
    else:
        return "High Risk", "red"


# ─────────────────────────────────────────────────────────────
# Display Results
# ─────────────────────────────────────────────────────────────

def print_results(features, risk_score):
    """Print a formatted results table to the terminal."""
    label, _ = get_risk_label(risk_score)
    print("\n" + "=" * 55)
    print("       MOTOR TEST — FEATURE EXTRACTION RESULTS")
    print("=" * 55)
    print(f"  {'Metric':<28} {'Value':>12}")
    print("-" * 55)
    print(f"  {'Taps Detected':<28} {features['num_taps']:>12d}")
    print(f"  {'Mean Amplitude (px)':<28} {features['mean_amplitude_px']:>12.1f}")
    print(f"  {'Normalized Amplitude (%)':<28} {features['norm_amplitude']:>12.1f}")
    print(f"  {'Amplitude Decay (%)':<28} {features['amplitude_decrement_pct']:>12.1f}")
    print(f"  {'Tap Frequency (Hz)':<28} {features['tap_frequency']:>12.2f}")
    print(f"  {'Mean Velocity (px/s)':<28} {features['mean_velocity']:>12.1f}")
    print(f"  {'Rhythm Variability (CoV)':<28} {features['rhythm_variability']:>12.3f}")
    print(f"  {'Hesitation Count':<28} {features['hesitation_count']:>12d}")
    print("-" * 55)
    print(f"  {'RISK SCORE':<28} {risk_score:>10.0f}/100")
    print(f"  {'ASSESSMENT':<28} {label:>12}")
    print("=" * 55)
    print("  ⚠  This is a screening tool, not a diagnosis.")
    print("     Consult a neurologist for clinical evaluation.\n")


def plot_results(times, distances, plot_data, features, risk_score):
    """Plot the distance-time graph with feature annotations."""
    label, color = get_risk_label(risk_score)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('NeuroSketch — Motor Test Analysis', fontsize=14, fontweight='bold')

    t_v = plot_data['t_valid']
    d_s = plot_data['d_smooth']
    peaks = plot_data['peak_idx']
    troughs = plot_data['trough_idx']
    pk_times = plot_data['peak_times']
    trend = plot_data['trend_line']

    # ── Top plot: signal + annotations ──
    ax1.plot(times, distances, color='lightgray', alpha=0.5, linewidth=0.8, label='Raw signal')
    ax1.plot(t_v, d_s, color='steelblue', linewidth=1.5, label='Smoothed signal')
    ax1.plot(t_v[peaks], d_s[peaks], 'v', color='red', markersize=8, label='Peaks (open)')
    ax1.plot(t_v[troughs], d_s[troughs], '^', color='green', markersize=8, label='Troughs (closed)')
    ax1.plot(pk_times, trend, '--', color='orangered', linewidth=2, label='Amplitude trend')

    ax1.set_ylabel('Distance (pixels)')
    ax1.set_title('Thumb–Index Finger Distance Over Time')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Bottom plot: feature summary card ──
    ax2.axis('off')
    summary_text = (
        f"Taps: {features['num_taps']}   |   "
        f"Frequency: {features['tap_frequency']:.2f} Hz   |   "
        f"Norm. Amplitude: {features['norm_amplitude']:.1f}%   |   "
        f"Decay: {features['amplitude_decrement_pct']:.1f}%\n"
        f"Velocity: {features['mean_velocity']:.1f} px/s   |   "
        f"Rhythm CoV: {features['rhythm_variability']:.3f}   |   "
        f"Hesitations: {features['hesitation_count']}"
    )
    ax2.text(0.5, 0.72, summary_text, transform=ax2.transAxes,
             fontsize=10, ha='center', va='center', family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray'))

    # Risk score badge
    ax2.text(0.5, 0.2, f"Risk Score: {risk_score:.0f}/100 — {label}",
             transform=ax2.transAxes, fontsize=14, fontweight='bold',
             ha='center', va='center', color='white',
             bbox=dict(boxstyle='round,pad=0.6', facecolor=color, edgecolor='darkgray'))

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
# Hand Detection (unchanged)
# ─────────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)


def detect_finger_landmarks(image):
    """Detect index finger tip and thumb tip landmarks."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    thumb_point = None
    index_point = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 8:  # Index finger tip
                    index_point = (cx, cy)
                    cv2.circle(image, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                elif id == 4:  # Thumb tip
                    thumb_point = (cx, cy)
                    cv2.circle(image, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    return image, thumb_point, index_point


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    if point1 is not None and point2 is not None:
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    else:
        return -1


# ─────────────────────────────────────────────────────────────
# Main — Capture & Analyze
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    capture_duration = 20  # seconds

    times = []
    distances = []

    print("\n🖐  MOTOR TEST — Finger Tapping")
    print("   Tap your index finger against your thumb")
    print("   as FAST and as BIG as possible.")
    print(f"   Recording for {capture_duration} seconds...\n")

    start_time = time.time()
    end_time = start_time + capture_duration

    while cap.isOpened() and time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            break

        frame, thumb, index = detect_finger_landmarks(frame)
        distance = calculate_distance(thumb, index)
        current_time = time.time() - start_time

        times.append(current_time)
        distances.append(distance)

        # On-screen display
        cv2.putText(frame, f'Distance: {distance:.2f} px',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        remaining = int(end_time - time.time())
        cv2.putText(frame, f'Time Left: {remaining} s',
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('NeuroSketch — Motor Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── Analyze ──
    result = extract_features(times, distances)

    if result is None:
        print("\n⚠  Not enough tapping data detected.")
        print("   Please ensure your hand is visible and you are")
        print("   tapping your thumb and index finger together.\n")

        # Fallback: show raw graph only
        plt.plot(times, distances)
        plt.title('Change in Distance over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Distance (pixels)')
        plt.grid(True)
        plt.show()
    else:
        features, plot_data = result
        risk_score = compute_risk_score(features)

        print_results(features, risk_score)
        plot_results(times, distances, plot_data, features, risk_score)
