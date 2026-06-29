import matplotlib.pyplot as plt
import pandas as pd

s_data = pd.DataFrame({
    's': [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 3, 4, 5],
    'mean_APD90': [166.95, 176.44, 205.96, 204.43, 204.14, 218.44, 209.68, 220.28, 205.71, 211.93],
})

x = s_data['s'].to_numpy()
y = s_data['mean_APD90'].to_numpy()

plt.figure(figsize=(6,4))
plt.plot(x, y, marker='o', linewidth=2)

plt.xlabel('$s$')
plt.ylabel('Mean APD90 (ms)')
plt.title('Mean APD90 vs $s$')
plt.grid(True)
plt.tight_layout()

plt.savefig('mean_APD90_vs_s.pdf', dpi=300)
plt.show()