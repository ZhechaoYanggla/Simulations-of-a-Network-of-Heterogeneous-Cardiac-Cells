#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import warnings

import myokit
import pints

warnings.filterwarnings("ignore")

# -----------------------------------
# Define the parameter names (8 of them)
# -----------------------------------
to_fit = ['NaCa', 'NaK', 'Clb', 'CaL', 'tos', 'K1', 'Ks', 'Kr']
parms_to_fit = ['I' + param for param in to_fit]

# -----------------------------------
# Define the Action Potential Model
# -----------------------------------
class ActionPotentialModel(pints.ForwardModel):
    parms = parms_to_fit

    def simulate(self, parameters, times):
        m, p, x = myokit.load("shannon_wang_puglisi_weber_bers_2004_a_p.mmt")
        m.var('reversal_potentials.E_Na_SL').set_rhs(-15.0)

        for index, par in enumerate(ActionPotentialModel.parms):
            m.var(par + '.p').set_rhs(parameters[index])

        s = myokit.Simulation(m, p)
        eps = 0.01
        d = s.run(times[-1] + eps, log_times=times)
        return np.asarray(d['cell.V'])

    def compute_quantities(self, parameters, times):
        m, p, x = myokit.load("shannon_wang_puglisi_weber_bers_2004_a_p.mmt")
        m.var('reversal_potentials.E_Na_SL').set_rhs(-15.0)

        for index, par in enumerate(ActionPotentialModel.parms):
            m.var(par + '.p').set_rhs(parameters[index])

        s = myokit.Simulation(m, p)
        eps = 0.01
        d = s.run(times[-1] + eps, log_times=times)

        return np.asarray([d['cell.V'], d['Ca_buffer.Cai']]), None

    def n_parameters(self):
        return len(ActionPotentialModel.parms)

# -----------------------------------
# Function to compute APD90
# -----------------------------------
def compute_apd90(voltage, time):
    peak_voltage = np.max(voltage)
    resting_voltage = np.min(voltage)
    threshold_voltage = resting_voltage + 0.7 * (peak_voltage - resting_voltage)

    indices = np.where(voltage >= threshold_voltage)[0]
    if len(indices) == 0:
        return None

    apd90_time_start = time[indices[0]]
    apd90_time_end = time[indices[-1]]
    apd90 = apd90_time_end - apd90_time_start

    return apd90

# -----------------------------------
# Generate from CSV instead of random
# -----------------------------------
def generate_until_n_good(model, times, csv_path="NR_after_drug_0.01kr.csv", n_good=800, apd90_threshold=400):
    # Read parameter values from CSV
    df_params = pd.read_csv(csv_path)
    columns = ['INaCa', 'INaK', 'IClb', 'ICaL', 'Itos', 'IK1', 'IKs', 'IKr']
    param_rows = df_params[columns].values

    good_params = []
    ap_traces = []
    total_attempts = 0

    for sampled_pars in param_rows:
        total_attempts += 1
        try:
            values, _ = model.compute_quantities(sampled_pars, times)
            AP = values[0]
            apd90 = compute_apd90(AP, times)

            if apd90 is not None and apd90 > apd90_threshold:
                good_params.append(sampled_pars.tolist())
                ap_traces.append(AP)
                print(f"[{total_attempts}] APD90 = {apd90:.2f} ? ({len(good_params)}/{n_good})")
            else:
                print(f"[{total_attempts}] APD90 = {apd90:.2f} ?")

            if len(good_params) >= n_good:
                break

        except Exception as e:
            print(f"[{total_attempts}] Simulation failed: {e}")
            continue

    print(f"\n? Collected {len(good_params)} valid samples after {total_attempts} total attempts.\n")
    return good_params, ap_traces

# -----------------------------------
# Main execution
# -----------------------------------
if __name__ == "__main__":
    # Time array
    times = np.linspace(0, 1500, 15000, endpoint=False)

    # Initialize the model
    model = ActionPotentialModel()

    # Generate using parameters from CSV
    good_params, ap_traces = generate_until_n_good(
        model,
        times,
        csv_path="NR_after_drug_0.01kr.csv",
        n_good=800,
        apd90_threshold=1400
    )

    # Save parameter sets
    df_good = pd.DataFrame(good_params, columns=parms_to_fit)
    df_good.to_csv("apd90_gt400_params_block.csv", index=False)
    print("Saved parameter sets to 'apd90_gt400_params.csv'")

    # Plot the traces
    plt.figure(figsize=(8, 6))
    for ap in ap_traces:
        plt.plot(times[400:-300], ap[400:-300], alpha=0.3, color='red')

    plt.title(f"AP traces with HR (n = {len(ap_traces)})")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Voltage (mV)")
    plt.ylim(-90, 60)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("apd90_gt400_traces_block.pdf", bbox_inches='tight')
    # plt.show()
