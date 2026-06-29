import matplotlib.pyplot as plt
import pandas as pd

g_data = pd.DataFrame({
    'g': [1, 3, 5, 7, 9],
    'mean_APD90': [209.36, 209.18, 207.21, 200.23, 193.20],
})

x = g_data['g'].to_numpy()
y = g_data['mean_APD90'].to_numpy()

plt.figure(figsize=(6, 4))
plt.plot(x, y, marker='o', linewidth=2)

plt.xlabel('$g$')
plt.ylabel('Mean APD90 (ms)')
plt.title('Mean APD90 vs $g$')
plt.grid(True)
plt.tight_layout()

plt.savefig('mean_APD90_vs_g.pdf', dpi=300)
plt.show()