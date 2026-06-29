import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Parameters
# ----------------------------------------------------------
N = 800
delta = 10
i = np.arange(N)
centre = N // 2
amplitude = 0.89

# ----------------------------------------------------------
# Sharpness values to study (observable effect)
# ----------------------------------------------------------
sharpness_values = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 1])

# ----------------------------------------------------------
# Local running average
# ----------------------------------------------------------
def local_avg(C, delta):
    kernel = np.ones(2 * delta + 1)
    return np.convolve(C, kernel, mode="same") / kernel.size

# ----------------------------------------------------------
# Fixed width configuration
# ----------------------------------------------------------
width = N // 3
i1, i2 = centre - width // 2, centre + width // 2

results = []

for s in sharpness_values:
    # Double-sigmoid probability profile
    sigmoid1 = 1 / (1 + np.exp(-(i - i1) * s))
    sigmoid2 = 1 / (1 + np.exp((i - i2) * s))
    p = amplitude * sigmoid1 * sigmoid2

    # Stochastic Bernoulli field
    C = (np.random.rand(N) < p).astype(int)

    # Local running average
    rho = local_avg(C, delta)

    results.append((s, p, C, rho))

# ----------------------------------------------------------
# Plot: 6 rows  3 columns
# ----------------------------------------------------------
fig, axs = plt.subplots(len(sharpness_values), 3, figsize=(14, 18), sharex=True)

for row, (s, p, C, rho) in enumerate(results):
    axs[row, 0].plot(p)
    axs[row, 0].set_ylabel(f"s = {s}")
    axs[row, 0].set_title("p(i)")

    axs[row, 1].stem(range(N), C, linefmt='gray', markerfmt=' ', basefmt=' ')
    axs[row, 1].set_title("C[i]")

    axs[row, 2].plot(rho)
    axs[row, 2].set_title(r"$\rho(i)$")

for ax in axs[-1, :]:
    ax.set_xlabel("i")

plt.tight_layout()
plt.savefig("double_sigmoid_sharpness_study_2.pdf", dpi=300, bbox_inches="tight")
plt.show()
