# AI Geothermal Optimization for double-flash plants
An Artificial Intelligence-Based Predictive Multi-Objective Optimization Framework for Double-Flash Geothermal Power Plants: Case of Olkaria Operating Conditions


# Project Description and Research Objectives

## 1. Project Background and Context
The Olkaria Geothermal Power Plant complex stands as a foundational anchor of Kenya's renewable energy mix. Currently, the operating infrastructure utilizes a **Single-Flash Geothermal Cycle**. In this configuration, high-temperature, high-pressure geo-fluid extracted from deep reservoirs is flashed down to a specific separation pressure. This process splits the resource into two phases: high-purity steam, which is routed directly to steam turbines to generate electrical power, and separated liquid brine, which is unexploited and immediately reinjected back into the subsurface reservoir.

While mechanically reliable, long-term operational profiles indicate that the plant frequently operates under volatile, **off-design conditions**. Variations in wellhead pressure, natural reservoir enthalpy decay, and shifting steam-to-brine ratios cause significant output drops. Furthermore, dumping the high-temperature separated brine straight into reinjection lines bypasses a massive amount of unexploited thermal energy. 

This project addresses this thermodynamic inefficiency by developing an **Artificial Intelligence-Based Predictive Multi-Objective Optimization Framework for a Double-Flash Geothermal Upgrade**. By integrating a secondary, low-pressure flash separator stage, the framework captures the residual enthalpy of the waste brine, producing an auxiliary steam stream to maximize resource efficiency. Because operational conditions fluctuate continuously, the system utilizes a data-driven **Artificial Neural Network (ANN)** as a physics-based surrogate model, coupled with an evolutionary **NSGA-II (Non-dominated Sorting Genetic Algorithm II)** optimization engine to execute dynamic control decisions in real time.

---

## 2. Process Architecture Schema
The operational framework maps the real-time physical parameters of the Olkaria units into a closed-loop intelligence and optimization sequence:



## 3. Project Objectives
## 3. Core Research Objectives

The project is structured around five sequential, highly interconnected technical objectives designed to bridge the gap between theoretical steady-state design and actual variable plant operations:

### Objective 1: Comprehensive Data Engineering and Statistical Baseline Profiling
* **Scope:** Clean, parse, split, and synchronize the multi-year raw industrial logging sheets for Generator Units 1, 2, and 3. Eliminate structural headers, resolve positional index shifts, and establish clean `datetime64` timelines.
* **Deliverable:** Quantify the exact capacity factor deficits, off-design volatility coefficients, and derated search boundaries across independent units to build a robust empirical baseline for model validation.

### Objective 2: Computational Thermodynamic Modeling and Exergy Derivation
* **Scope:** Build a high-fidelity mathematical and physical simulation model of the single-flash and expanded double-flash geothermal cycles in Python. Integrate international formulation standard steam tables (`pyXSteam` / `iapws`) to calculate precise enthalpy ($h$), entropy ($s$), and irreversibility deltas ($I = T_0 \cdot S_{gen}$).
* **Deliverable:** Mathematically derive and calculate the plant's true continuous **Exergy Efficiency ($\eta_{ex}$)** to serve as the ultimate optimization target variable alongside raw electrical power output.

### Objective 3: Multi-Output Neural Network (ANN) Surrogate Model Development
* **Scope:** Design and train an optimized feedforward Multi-Layer Perceptron (MLP) Neural Network for each generator. The network acts as a data-driven surrogate model, learning the complex, non-linear thermodynamic mapping from input operating conditions (mass flows, temperatures, pressures) to output metrics.
* **Deliverable:** Replace expensive, computationally heavy thermodynamic equations with a sub-millisecond neural network forward pass capable of predicting power output ($MW$) and exergy efficiency ($\%$) with high precision ($R^2 > 0.95$).

### Objective 4: NSGA-II Multi-Objective Optimization and Pareto Frontier Generation
* **Scope:** Implement the Non-dominated Sorting Genetic Algorithm II (NSGA-II) using the trained ANN surrogate as the multi-objective fitness evaluation function. Define the decision variable boundaries (HP/LP separator pressures, turbine chest pressure, inlet flow rates) based on observed derated states.
* **Deliverable:** Generate a dense, well-distributed **Pareto-Optimal Frontier** representing the optimal trade-offs between maximizing total net power output ($MW$) and maximizing plant exergy efficiency ($\%$). This frontier serves as the logical backbone for dynamic, adaptive low-pressure flash switching control.

### Objective 5: Predictive Maintenance and Multi-Component Fault Diagnostics
* **Scope:** Develop independent machine learning diagnostic pipelines (utilizing sequential LSTM networks or unsupervised Autoencoders) to track and identify localized equipment degradation signals.
* **Deliverable:** Detect early-warning operational thresholds for (a) steam purity degradation via real-time conductivity tracking to mitigate turbine blade scaling, (b) inlet nozzle blockages via absolute left-hand/right-hand temperature divergence calculations ($|LH - RH|$), and (c) condenser performance decay via exhaust vacuum back-pressure drifts.