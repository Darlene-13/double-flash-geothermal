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
1. To develop a reference thermodynamic model for a double-flash geothermal power plant under off-design operating conditions 
2. To optimize operating parameters of the developed model using multi-objective optimization framework for optimal net power output and exergy efficiency.
3. To compare the baseline plant operation with numerical simulation data under variable operating conditions. 
