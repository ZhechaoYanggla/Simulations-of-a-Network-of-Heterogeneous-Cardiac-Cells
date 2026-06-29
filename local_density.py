import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Parameters
# ----------------------------------------------------------
N = 800
delta = 10
i = np.arange(N)
centre = N // 2
amplitude = 0.5
sharpness = 1 / (N * 1e-5)

# ----------------------------------------------------------
# Local running average
# ----------------------------------------------------------
def local_avg(C, delta):
    N = len(C)
    rho = np.zeros(N)
    for k in range(N):
        kmin = max(0, k - delta)
        kmax = min(N - 1, k + delta)
        rho[k] = np.sum(C[kmin:kmax + 1]) / (kmax - kmin + 1)
    return rho

# ----------------------------------------------------------
# Two double-sigmoid configurations
# ----------------------------------------------------------
configs = [
    ("w = N//3", N // 3),
    ("w = 2N", N * 2),
]

results = []

for label, width in configs:
    i1, i2 = centre - width // 2, centre + width // 2
    sigmoid1 = 1 / (1 + np.exp(-(i - i1) * sharpness))
    sigmoid2 = 1 / (1 + np.exp((i - i2) * sharpness))
    p = amplitude * sigmoid1 * sigmoid2

    C = (np.random.rand(N) < p).astype(int)
    rho = local_avg(C, delta)

    results.append((label, p, C, rho))

# ----------------------------------------------------------
# Plot: 2 rows  3 columns
# ----------------------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(14, 6), sharex=True)

for row, (label, p, C, rho) in enumerate(results):
    axs[row, 0].plot(p)
    axs[row, 0].set_ylabel(label)
    axs[row, 0].set_title("p(i)")

    axs[row, 1].stem(range(N), C, linefmt='gray', markerfmt=' ', basefmt=' ')
    axs[row, 1].set_title("C[i]")

    axs[row, 2].plot(rho)
    axs[row, 2].set_title(r"$\rho(i)$")

for ax in axs[-1, :]:
    ax.set_xlabel("i")

plt.tight_layout()
plt.savefig("double_sigmoid_diagnostics.pdf", dpi=300, bbox_inches="tight")

plt.show()
