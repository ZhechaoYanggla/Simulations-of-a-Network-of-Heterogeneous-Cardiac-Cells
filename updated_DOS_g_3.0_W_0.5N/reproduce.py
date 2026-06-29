#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import myokit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# -----------------------------
# Parameters
# -----------------------------
param_cols = ['INaCa', 'INaK', 'IClb', 'ICaL', 'Itos', 'IK1', 'IKs', 'IKr']
N = 800
g = 3.0
t = 500

# -----------------------------
# Gap junctions: ring topology
# -----------------------------
connections = [(k, (k+1) % N, g) for k in range(N)]

# -----------------------------
# APD90 calculation
# -----------------------------
def compute_apd90_single(voltage, time):
    peak_voltage = np.max(voltage)
    resting_voltage = np.min(voltage)
    threshold = resting_voltage + 0.1 * (peak_voltage - resting_voltage)
    idx = np.where(voltage >= threshold)[0]
    if len(idx) == 0:
        return None
    return time[idx[-1]] - time[idx[0]]

# -----------------------------
# Corona ring plot
# -----------------------------
def plot_corona_ring(data_log, time_array, n_cells, title='', save_path=None):
    time_array = np.array(time_array)
    mask = (time_array >= 20) & (time_array <= 500)
    time_array_plot = time_array[mask]

    V_matrix = np.zeros((len(time_array_plot), n_cells))
    for idx in range(n_cells):
        key = f"{idx}.cell.V"
        V_matrix[:, idx] = np.array(data_log[key])[mask]

    V_matrix = np.hstack([V_matrix, V_matrix[:, 0:1]])
    vmax = np.nanmax(V_matrix)
    vmin = np.nanmin(V_matrix)

    theta = np.linspace(0, 2*np.pi, n_cells+1)
    r = time_array_plot - time_array_plot[0]
    Theta, R = np.meshgrid(theta, r)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    ax.grid(False)
    ax.pcolormesh(Theta, R, V_matrix, cmap='seismic', shading='auto', vmin=vmin, vmax=vmax)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if title:
        ax.set_title(title, va='bottom')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved corona ring heatmap: {save_path}")
    plt.close()

# -----------------------------
# Running average
# -----------------------------
def local_avg(C, delta):
    N = len(C)
    rho = np.zeros(N)
    for k in range(N):
        kmin = max(0, k - delta)
        kmax = min(N - 1, k + delta)
        window = C[kmin:kmax + 1]
        rho[k] = np.sum(window) / len(window)
    return rho

# -----------------------------
# Load cell parameters
# -----------------------------
df_params = pd.read_csv('params_attempt28.csv')  # use your CSV
assert len(df_params) == N, "CSV does not have the expected number of cells"

# -----------------------------
# Load model and protocol
# -----------------------------
model, protocol, _ = myokit.load('shannon_wang_puglisi_weber_bers_2004_a_p.mmt')
model.get('cell.V').set_label('membrane_potential')

# -----------------------------
# Run simulation
# -----------------------------
try:
    sim = myokit.SimulationOpenCL(model, protocol, ncells=N)
except Exception:
    print("[INFO] OpenCL unavailable, running on CPU")
    sim = myokit.Simulation(model, protocol)

for col in param_cols:
    sim.set_field(f'{col}.p', df_params[col].values)

sim.set_constant('cell.stim_amplitude', 9.5)
sim.set_paced_cells(nx=N, x=0)
sim.set_connections(connections)

# Log all cells for corona plot
log_list = ['environment.time'] + [f'{i}.cell.V' for i in range(N)]
d = sim.run(t, log=log_list)
time_array = d['environment.time']

# -----------------------------
# Corona ring plot
# -----------------------------
#plot_corona_ring(d, time_array, N, title='Corona Ring Attempt 16', save_path='ap_circular_heatmap_attempt16.png')

# -----------------------------
# Compute APD90
# -----------------------------
apd_array = np.array([compute_apd90_single(d[f'{idx}.cell.V'], time_array) for idx in range(N)])

# -----------------------------
# Running average plot
# -----------------------------
delta = 3
apd_running_avg = local_avg(apd_array, delta)
plt.figure(figsize=(10,6))
plt.plot(np.arange(N), apd_running_avg, marker='o')
plt.xlabel("Cell Index")
plt.ylabel(f"Running Avg APD90 (delta={delta})")
plt.title("Running Average of APD90 Along Ring")
plt.grid(True)
plt.tight_layout()
#plt.savefig("APD90_running_avg_attempt16.png", dpi=300)
plt.close()
print("[INFO] Running average plot saved")


# Create combined figure
# -----------------------------
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])  # Left: corona, Right: running avg

# -----------------------------
# Left: Corona Ring
# -----------------------------
ax0 = fig.add_subplot(gs[0], polar=True)
time_array = np.array(time_array)
time_mask = (time_array >= 20) & (time_array <= 500)
time_plot = time_array[time_mask]

V_matrix = np.zeros((len(time_plot), N))
for idx in range(N):
    key = f"{idx}.cell.V"
    V_matrix[:, idx] = np.array(d[key])[time_mask]

V_matrix = np.hstack([V_matrix, V_matrix[:, 0:1]])  # wrap-around
vmax = np.nanmax(V_matrix)
vmin = np.nanmin(V_matrix)

theta = np.linspace(0, 2*np.pi, N+1)
r = time_plot - time_plot[0]
Theta, R = np.meshgrid(theta, r)

c0 = ax0.pcolormesh(Theta, R, V_matrix, cmap='seismic', shading='auto', vmin=vmin, vmax=vmax)
ax0.set_theta_direction(-1)
ax0.set_theta_zero_location("N")
ax0.set_xticklabels([])
ax0.set_yticklabels([])
#ax0.set_title("Corona Ring", va='bottom')
#fig.colorbar(c0, ax=ax0, label="Voltage (mV)", orientation='vertical')

# -----------------------------
# Right: Running Average
# -----------------------------
ax1 = fig.add_subplot(gs[1])
ax1.plot(np.arange(N), apd_running_avg, marker='o')
ax1.set_xlabel("Cell Index")
ax1.set_ylabel(f"Running Avg APD90 (ms)")
#ax1.set_title("Running Average of APD90")
ax1.grid(True)

# -----------------------------
# Save combined figure
# -----------------------------
plt.tight_layout()
plt.savefig("combined_corona_running_avg.pdf", dpi=300)
plt.savefig("combined_corona_running_avg.png", dpi=300)
plt.close()
print("[INFO] Combined figure saved: combined_corona_running_avg.png")