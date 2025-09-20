# -*-coding:utf-8-*-
import numpy as np


def interpolation(base_line, the_count):
    base_line_length = len(base_line)

    if base_line[base_line_length - 1] == 0:
        base_line[base_line_length - 1] = the_count

    while np.count_nonzero(base_line == 0) > 0:
        zero_location = np.where(base_line == 0)[0]
        max_number = len(zero_location)


        if zero_location[0] - 1 < 1:
            base_line[zero_location[0]] = the_count
        else:
            base_line[zero_location] = np.interp(zero_location,
                                                 [zero_location[0] - 1, zero_location[max_number - 1] + 1],
                                                 [base_line[zero_location[0] - 1],
                                                  base_line[zero_location[max_number - 1] + 1]])

    output = base_line
    return output


def bad_value_processing(input):
    The_length_window = 500
    bad_length = len(input)
    bad_length = (
                             bad_length // The_length_window) * The_length_window
    for ii in range(0, bad_length, The_length_window):
        mean_signal = sum(input[ii:ii + The_length_window]) / The_length_window
        std_signal = np.std(input[ii:ii + The_length_window])
        Bad_value = np.where((input[ii:ii + The_length_window] > (mean_signal + 3 * std_signal)) | (
                    input[ii:ii + The_length_window] < (mean_signal - 3 * std_signal)))
        input[Bad_value[0] + ii - 1] = 0
    mean_input = np.mean(input)
    output = interpolation(input, mean_input)
    return output


import numpy as np


def SSA(signal, windowLen):
    # Step 1: Build trajectory matrix
    N = len(signal)
    if windowLen > N / 2:
        windowLen = N - windowLen
    K = N - windowLen + 1
    X = np.zeros((windowLen, K))
    for i in range(K):
        X[:, i] = signal[i:i + windowLen]

    # Step 2: Singular Value Decomposition
    S = np.dot(X, X.T)
    U, autoval, _ = np.linalg.svd(S)
    U = U[:, :K]
    V = np.dot(X.T, U)

    # Step 3: Grouping
    I = np.arange(0, windowLen // 4)
    Vt = V.T
    rca = np.dot(U[:, I], Vt[I, :])

    # Step 4: Reconstruction
    y = np.zeros(N)
    Lp = min(windowLen, K)
    Kp = max(windowLen, K)

    # Reconstruction 1~Lp-1
    for k in range(Lp - 1):
        for m in range(k + 1):
            y[k] += (1 / (k + 1)) * rca[m, k - m]

    # Reconstruction Lp~Kp
    for k in range(Lp - 1, Kp - 1):
        for m in range(Lp):
            y[k] += (1 / Lp) * rca[m, k - m]

    # Reconstruction Kp+1~N
    for k in range(Kp - 1, N - 1):
        for m in range(k - Kp + 2, N - Kp + 1):
            y[k] += (1 / (N - k)) * rca[m, k - m]

    signalFiltered = y
    return signalFiltered

if __name__ == "__main__":
    import numpy as np

    # 1. Generate a simulated UC signal (with missing values and anomalies)
    n = 2000
    t = np.linspace(0, 6 * np.pi, n)
    uc_raw = 50 + 10 * np.sin(t) + np.random.normal(0, 0.8, n)
    uc_raw[100:120] = 0  # missing segment
    uc_raw[800:805] = 120  # abnormal high values
    uc_raw[1500:1510] = -10  # abnormal low values

    # 2. Apply preprocessing step by step
    mean_uc = np.mean(uc_raw)
    uc_interp = interpolation(uc_raw.copy(), mean_uc)
    uc_clean = bad_value_processing(uc_interp.copy())
    uc_final = SSA(uc_clean.copy(), windowLen=30)

    # 3. Print summary statistics before/after
    print("Raw signal: min={:.2f}, max={:.2f}".format(uc_raw.min(), uc_raw.max()))
    print("Processed signal: min={:.2f}, max={:.2f}".format(uc_final.min(), uc_final.max()))
