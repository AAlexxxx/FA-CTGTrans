import numpy as np


def preprocess_signal(fhr_signal, fs=4):
    """
    Preprocess the FHR signal by removing unreliable parts and interpolating missing values.

    Parameters:
        fhr_signal (array-like): The input FHR signal.

    Returns:
        Interpolated FHR signal (array-like)
    """
    # Remove small parts from the signal
    fhr_processed = remove_unreliable_parts(fhr_signal.copy())

    # Interpolate missing values in the signal
    fhr_interpolated = interpolate_fhr(fhr_processed)

    return fhr_interpolated


def remove_unreliable_parts(fhr_signal, short_gap_threshold=20, anomaly_threshold=25):
    """
    Clean the FHR signal by replacing invalid values, short gaps, and anomalies with zeros.
    Parameters:
        fhr_signal (array-like): The input FHR signal.
        short_gap_threshold (int): Threshold in samples for short gaps to be removed (default is 20 samples).短间隔阈值
        anomaly_threshold (int): Threshold in FHR change to detect anomalies (default is 25 bpm).异常阈值

    Returns:
        array-like: The cleaned FHR signal.
    """
    fhr = fhr_signal.copy()

    # Remove values outside the valid range
    fhr[(fhr < 50) | (fhr > 220)] = 0

    # Identify gaps in the signal fhr[1:]
    gap_starts = np.where((fhr[:-1] == 0) & (fhr[1:] > 0))[0] + 1

    # Remove short gaps
    for start in gap_starts:
        gap_end = np.where(fhr[start:] == 0)[0]
        if gap_end.size > 0 and gap_end[0] < short_gap_threshold:
            fhr[start:start + gap_end[0] + 1] = 0

    # Detect and remove anomalies (e.g., doubling or halving patterns)
    for start in gap_starts:
        gap_end = np.where(fhr[start:] == 0)[0]
        if gap_end.size > 0 and gap_end[0] < 30 * 4:  # Longer gaps
            gap_length = gap_end[0]
            prev_valid = np.where(fhr[:start] > 0)[0]
            next_valid = np.where(fhr[start + gap_length:] > 0)[0]

            if prev_valid.size > 0 and next_valid.size > 0:
                prev_value = fhr[prev_valid[-1]]
                next_value = fhr[start + gap_length + next_valid[0]]

                # Remove doubling or halving anomalies
                if (fhr[start] - prev_value < -anomaly_threshold) and (
                        fhr[start + gap_length - 1] - next_value < -anomaly_threshold):
                    fhr[start:start + gap_length] = 0
                elif (fhr[start] - prev_value > anomaly_threshold) and (
                        fhr[start + gap_length - 1] - next_value > anomaly_threshold):
                    fhr[start:start + gap_length] = 0

    return fhr


def interpolate_fhr(fhr_signal):
    """
    Interpolate missing values (zeros) in the FHR signal.

    Parameters:
        fhr_signal (array-like): The input FHR signal.

    Returns:
        array-like: The interpolated FHR signal.
    """
    fhr = fhr_signal.copy()

    # Flatten the signal if it's multidimensional
    if fhr.ndim > 1:
        fhr = fhr.flatten()

    # Find the indices of valid (non-zero) values
    valid_indices = np.where(fhr > 0)[0]

    if valid_indices.size > 0:
        # Set initial zeros to the first valid value
        first_valid = valid_indices[0]
        fhr[:first_valid] = fhr[first_valid]

        # Interpolate over zero regions
        idx = first_valid
        while idx is not None and idx < len(fhr):
            zero_start = np.where(fhr[idx:] == 0)[0]
            if zero_start.size == 0:
                break
            zero_start = zero_start[0] + idx

            next_valid = np.where(fhr[zero_start:] > 0)[0]
            if next_valid.size == 0:
                break
            next_valid = next_valid[0] + zero_start

            # Linear interpolation for zero region (Fix: Remove +1)
            fhr[zero_start:next_valid] = np.linspace(fhr[zero_start - 1], fhr[next_valid], next_valid - zero_start)
            # fhr[zero_start-1:next_valid+1] = np.linspace(fhr[zero_start - 1], fhr[next_valid], next_valid - zero_start+2)

            idx = next_valid

        # Set trailing zeros to the last valid value
        last_valid = valid_indices[-1]
        fhr[last_valid:] = fhr[last_valid]

    return fhr

if __name__ == "__main__":
    # Example FHR signal with missing values and anomalies
    fhr_raw = np.array(
        [120, 125, 130, 0, 0, 0, 140, 145, 230, 150, 0, 0, 155, 160, 0, 0, 0, 0, 170, 175]
    )

    # Run preprocessing
    fhr_clean = preprocess_signal(fhr_raw)

    # Print results
    print("Raw FHR signal:", fhr_raw)
    print("Processed FHR signal:", fhr_clean)

