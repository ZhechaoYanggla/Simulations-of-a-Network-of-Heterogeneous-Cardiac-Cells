import matplotlib.pyplot as plt
import pandas as pd

g3 = pd.DataFrame({
    'A': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'mean_APD90': [156.00, 160.03, 162.83, 170.68, 178.06,
                   181.57, 196.27, 222.48, 275.89],
})

x = g3['A'].to_numpy()
y = g3['mean_APD90'].to_numpy()

plt.figure(figsize=(6, 4))
plt.plot(x, y, marker='o', linewidth=2)

plt.xlabel('$A$')
plt.ylabel('Mean APD90 (ms)')
plt.title(r'Mean APD90 vs $A$ ($g = 3$)')
plt.grid(True)
plt.tight_layout()

plt.savefig('mean_APD90_vs_amplitude_g3.pdf', dpi=300)
plt.show()
