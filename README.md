# Simulations-of-a-Network-of-Heterogeneous-Cardiac-Cells
Simulations of a Network of Heterogeneous Cardiac Cells

# Cardiac Network Simulation with Hyper Responders

This repository contains the code accompanying the paper

**"Spatial Distribution of Hyper Responders in Heterogeneous Cardiac Cell Networks"**

The code investigates how the spatial distribution of hyper responder (HR) cells and intercellular coupling influence electrical activity in heterogeneous cardiac tissue.

## Overview

The workflow consists of two main stages:

1. Generate hyper responder (HR) cell models from experimentally calibrated normal responder (NR) models.
2. Simulate a coupled cardiac cell network containing both HR and NR cells.

---

# Requirements

* Python 3.10+
* Myokit
* NumPy
* Pandas
* Matplotlib


Install the required packages using

```bash
pip install numpy pandas matplotlib myokit 
```

---


# Step 1. Generate Hyper Responders

Run

```
python par_generate_block.py
```

This script

* loads experimentally calibrated normal responder parameter sets,
* simulates action potentials,
* computes APD90,
* identifies cells with prolonged repolarization,
* classifies these as hyper responders (HRs),
* saves the selected parameter sets.

### Input

```
NR_after_drug_0.01kr.csv
```

### Output

```
apd90_gt400_params_block.csv
apd90_gt400_traces_block.pdf
```

---

# Step 2. Run Network Simulations

Run

```
python shuffle_nopre.py
```

This script

* constructs a one-dimensional ring of 800 electrically coupled cells,
* assigns each cell as either a normal responder (NR) or hyper responder (HR),
* generates a spatial HR distribution using a double-sigmoid probability profile,
* performs tissue-level simulations using Myokit,
* computes APD90 for every cell,
* produces summary figures.

### Input

```
apd90_gt400_params_block.csv
NR_ones.csv
```

### Output

* APD90 histogram
* Action potential traces
* Circular voltage heatmap
* Cell parameter table
* Simulation metadata

---

# Model

Simulations use the Shannon–Wang–Puglisi–Weber–Bers rabbit ventricular action potential model implemented in Myokit.

---


# Contact

Zhechao Yang

School of Mathematics and Statistics

University of Glasgow
