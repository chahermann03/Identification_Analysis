# 🌀 Cosmic Void Statistics — ML vs Physical Simulations

This repository provides all computational analysis for my Bachelor's thesis:  
**"Cosmic Void Statistics in Comparison of Machine Learning and Physical Simulations"**

It contains all scripts and workflows required to process simulation data and perform statistical analyses on cosmic voids using three approaches:  
**2LPT**, **N-body**, and a **Field-level emulator**.

---

##  Pipeline Overview

1. ### **Data Conversion**
   - The raw binary input files containing tracer velocities and positions are converted to readable ASCII format.
   - This step is handled by the `*_read.py` scripts.

2. ### **Void Identification (Sparkling)**
   - The converted data is passed to **Sparkling**, a void finder.
   - Execution is initialized via `identyfication.sh`, which launches an array of jobs on the Raven cluster.
   - Each job runs with a specific configuration in `sparkling_box.param`, enabling void finding for each realization.
   - The script supports all simulation types: **2LPT**, **N-body**, and **emulator**.

3. ### **Void Statistics**
   - Once voids are identified and saved, the main analysis pipeline computes:
     - **Void Size Function (VSF)**
     - **Density Profiles**
     - **Velocity Profiles**
   - Separate scripts exist for each simulation type:
     - `*_pipeline.py` for 2LPT, N-body, and emulator respectively.

4. ### **Comparative Analysis**
   - Comparative plots and residual statistics between simulation types are generated using:
     - `comparison_nbody+emu.py`
     - `comparison_nbody+2lpt.py`
   - These scripts produce side-by-side comparisons and visualizations for:
     - Void size distributions
     - Profile shapes
     - Statistical differences
