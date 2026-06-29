#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import myokit
import matplotlib.pyplot as plt
import os

# -----------------------------
# Parameters
# -----------------------------
param_cols = ['INaCa', 'INaK', 'IClb', 'ICaL', 'Itos', 'IK1', 'IKs', 'IKr']

N = 800
g = 7.0
t = 500
max_attempts = 10000

i = np.arange(N)  # needed for sigmoid calculation

# -----------------------------
# Load single-cell phenotypes
# -----------------------------
df_hr = pd.read_csv('apd90_gt400_params_block.csv')
df_nr = pd.read_csv('NR_ones.csv')

one_hr = df_hr.iloc[0].copy()
one_nr = df_nr.iloc[0].copy()

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
# AP traces
# -----------------------------
def plot_ap_traces(data_log, time_array, cell_indices, title='', save_path=None):
    plt.figure(figsize=(8, 5))
    for idx in cell_indices:
        plt.plot(time_array, data_log[f'{idx}.cell.V'], label=f'Cell {idx}')
    plt.title(title or 'Action Potential Traces')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Saved AP traces: {save_path}")
    plt.close()

# ======================================================
#       SINGLE SIMULATION
# ======================================================

for attempt in range(1, max_attempts + 1):
    print(f"\n--- Attempt {attempt} ---")

    seed = np.random.randint(0, 1_000_000)
    np.random.seed(seed)

    # -----------------------------
    # Double-opposing sigmoid steps (hardcoded)
    # -----------------------------
    sharpness = 1/(N * 1e-5)
    width = N//3
    centre = N // 2
    amplitude = 0.89

    i1, i2 = (centre - width // 2, centre + width // 2)

    sigmoid1 = 1 / (1 + np.exp(-(i - i1) * sharpness))   # rising
    sigmoid2 = 1 / (1 + np.exp((i - i2) * sharpness))    # falling

    p_double = amplitude * sigmoid1 * sigmoid2
    C = (np.random.rand(N) < p_double).astype(int)

    print(f"[INFO] HR fraction = {C.mean():.3f}, seed = {seed}")

    df_combined = pd.DataFrame(columns=param_cols)
    for col in param_cols:
        df_combined[col] = np.where(C == 1, one_hr[col], one_nr[col])
    df_combined['group'] = np.where(C == 1, 'HR', 'NR')

    try:
        model, protocol, _ = myokit.load('shannon_wang_puglisi_weber_bers_2004_a_p.mmt')
        model.get('cell.V').set_label('membrane_potential')

        try:
            sim = myokit.SimulationOpenCL(model, protocol, ncells=N)
        except Exception:
            print("[INFO] OpenCL unavailable, running on CPU")
            sim = myokit.Simulation(model, protocol)

        for col in param_cols:
            sim.set_field(f'{col}.p', df_combined[col].values)

        sim.set_constant('cell.stim_amplitude', 9.5)
        sim.set_paced_cells(nx=N, x=0)
        sim.set_connections(connections)

        d = sim.run(t, log=['environment.time', 'cell.V', 'Ca_buffer.Cai', 'cell.diffusion_current'])
        print("[SUCCESS] Simulation completed")

        time = d['environment.time']
        apd_array = np.array([compute_apd90_single(d[f'{idx}.cell.V'], time) for idx in range(N)])
        df_combined['APD90'] = apd_array
        df_combined['sharpness'] = sharpness
        df_combined['width'] = width
        df_combined['amplitude'] = amplitude

        mean_total = np.nanmean(apd_array)
        mean_hr = df_combined[df_combined['group']=='HR']['APD90'].mean()
        mean_nr = df_combined[df_combined['group']=='NR']['APD90'].mean()

        print(f"[APD90] Mean={mean_total:.2f}, HR_mean={mean_hr:.2f}, NR_mean={mean_nr:.2f}")

        # -----------------------------
        # Outputs
        # -----------------------------
        plt.figure(figsize=(6,4))
        plt.hist(apd_array[~np.isnan(apd_array)], bins=30, color='skyblue', edgecolor='k')
        plt.axvline(mean_total, color='red', linestyle='--', linewidth=2)
        plt.title(f'APD90 Distribution\nsharpness={sharpness:.4f}, width={width}, amplitude={amplitude}')
        plt.xlabel('APD90 (ms)')
        plt.ylabel('Count')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'apd90_hist_attempt{attempt}.pdf')
        plt.close()

        plot_ap_traces(
            d, time, list(range(N)),
            title=f'AP Traces\nsharpness={sharpness:.4f}, width={width}, amplitude={amplitude}',
            save_path=f'ap_traces_attempt{attempt}.pdf'
        )

        plot_corona_ring(
            d, time, N,
            title=f'Corona Ring\nsharpness={sharpness:.4f}, width={width}, amplitude={amplitude}',
            save_path=f'ap_circular_heatmap_attempt{attempt}.png'
        )

        df_combined.to_csv(f'params_attempt{attempt}.csv', index=False)

        with open(f'metadata_attempt{attempt}.txt', 'w') as f:
            f.write(f"Attempt: {attempt}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"HR fraction: {C.mean():.3f}\n")
            f.write(f"Mean APD90: {mean_total:.2f}\n")
            f.write(f"Mean HR APD90: {mean_hr:.2f}\n")
            f.write(f"Mean NR APD90: {mean_nr:.2f}\n")
            f.write(f"Sharpness: {sharpness:.4f}\n")
            f.write(f"Width: {width}\n")
            f.write(f"Amplitude: {amplitude}\n")

        break

    except Exception as e:
        print(f"[FAIL] Attempt {attempt}: {e}")

else:
    print(f"[ERROR] All attempts failed.")
