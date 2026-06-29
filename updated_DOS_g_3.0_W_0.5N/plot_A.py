import matplotlib.pyplot as plt
import pandas as pd

A_data = pd.DataFrame({
    'A': [0.5, 0.6, 0.7, 0.8, 0.83, 0.85, 0.87, 0.89],
    'mean_APD90': [163.37, 171.60, 179.58, 184.97, 188.04, 189.99, 198.13, 209.18],
})

x = A_data['A'].to_numpy()
y = A_data['mean_APD90'].to_numpy()

plt.figure(figsize=(6,4))
plt.plot(x, y, marker='o', linewidth=2)

plt.xlabel('$A$')
plt.ylabel('Mean APD90 (ms)')
plt.title('Mean APD90 vs $A$')
plt.grid(True)
plt.tight_layout()

plt.savefig('mean_APD90_vs_A.pdf', dpi=300)
plt.show()