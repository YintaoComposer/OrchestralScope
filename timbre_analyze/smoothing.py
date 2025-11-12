import numpy as np


def hmm_smooth_labels(labels: np.ndarray, num_classes: int, self_bias: float = 0.6, eps: float = 1e-4) -> np.ndarray:
    T = np.full((num_classes, num_classes), (1.0 - self_bias) / max(1, num_classes - 1))
    np.fill_diagonal(T, self_bias)
    logT = np.log(T + 1e-12)

    # Observation: given original label, set probability for that class to 1-eps, distribute eps evenly among others
    N = len(labels)
    logB = np.full((N, num_classes), np.log(eps / max(1, num_classes - 1)))
    for t, lab in enumerate(labels):
        logB[t, lab] = np.log(1.0 - eps)

    # Initial uniform
    logpi = np.full(num_classes, -np.log(num_classes))

    # Viterbi
    dp = np.zeros((N, num_classes))
    ptr = np.zeros((N, num_classes), dtype=int)
    dp[0] = logpi + logB[0]
    for t in range(1, N):
        for j in range(num_classes):
            prev = dp[t - 1] + logT[:, j]
            ptr[t, j] = np.argmax(prev)
            dp[t, j] = prev[ptr[t, j]] + logB[t, j]
    out = np.zeros(N, dtype=int)
    out[-1] = int(np.argmax(dp[-1]))
    for t in range(N - 2, -1, -1):
        out[t] = ptr[t + 1, out[t + 1]]
    return out


