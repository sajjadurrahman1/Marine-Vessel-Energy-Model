"""
Ship Energy System Simulation & Emission Optimization
Author: <Your Name>
Description:
    Demo project modelling a simplified ship energy system (propulsion, electrical and thermal),
    and optimizing speed to minimize fuel consumption and CO2 emissions for a given voyage.
    Uses only synthetic (demo) data.

How to run:
    python ship_energy_model.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Demo input data & constants
# -----------------------------

# Voyage parameters (demo)
DISTANCE_NM = 1000.0  # nautical miles
TIME_WINDOW_H = 60.0  # must arrive within 60 hours

# Speed limits
MIN_SPEED_KTS = 10.0
MAX_SPEED_KTS = 20.0

# Fuel lower heating value (just for concept, not used in detail)
LHV_HFO = 40e6  # J/kg (simplified)

# Emission factors (ton CO2 per ton fuel) – demo values
EMISSION_FACTOR_HFO = 3.114  # tCO2 / t fuel (approx)
EMISSION_FACTOR_LNG = 2.75   # tCO2 / t fuel (demo)
EMISSION_FACTOR_BIO = 0.3    # tCO2 / t fuel (assuming low net CO2, demo)


# -----------------------------
# 2. Marine system models (simplified)
# -----------------------------

def propulsive_power_kw(speed_knots: float) -> float:
    """
    Very simplified propulsion model:
    Power ~ a * v^3 + hotel offset

    speed_knots: ship speed [knots]
    returns: propulsive power [kW]
    """
    # Convert to m/s (not strictly needed, but realistic)
    speed_ms = speed_knots * 0.51444

    # Resistance / power approximation constant (demo!)
    a = 0.8  # kW / (m/s)^3, arbitrary tuned constant

    power_kw = a * speed_ms**3
    return power_kw


def main_engine_sfc_g_per_kwh(power_fraction: float) -> float:
    """
    Specific fuel consumption curve for main engine.
    Simplified: best efficiency at ~80% load, worse at low/high loads.

    power_fraction: engine load fraction [0..1]
    returns: SFC [g/kWh]
    """
    # Base SFC at optimal load (demo value)
    base = 170.0  # g/kWh

    # Quadratic penalty away from 0.8 load (demo)
    penalty = 60.0 * (power_fraction - 0.8) ** 2
    return base + penalty


def fuel_consumption_t_per_h(power_kw: float, rated_power_kw: float) -> float:
    """
    Calculate hourly fuel consumption of main engine.

    power_kw: actual power [kW]
    rated_power_kw: rated engine power [kW]
    returns: fuel [ton/h]
    """
    if power_kw <= 0:
        return 0.0

    load_fraction = min(max(power_kw / rated_power_kw, 0.05), 1.0)
    sfc_g_per_kwh = main_engine_sfc_g_per_kwh(load_fraction)
    fuel_kg_per_h = sfc_g_per_kwh * power_kw / 1000.0  # g/kWh -> kg/h
    fuel_t_per_h = fuel_kg_per_h / 1000.0
    return fuel_t_per_h


def electrical_load_kw(scenario: str = "cruise") -> float:
    """
    Hotel and service electrical load.

    scenario: "port", "maneuvering", "cruise"
    returns: load [kW]
    """
    if scenario == "port":
        return 1500.0
    elif scenario == "maneuvering":
        return 2500.0
    else:  # "cruise"
        return 3000.0


def thermal_load_kw(scenario: str = "cruise") -> float:
    """
    Thermal demand for heating, hot water, etc.

    scenario: "port", "maneuvering", "cruise"
    returns: load [kW]
    """
    if scenario == "port":
        return 1000.0
    elif scenario == "maneuvering":
        return 1500.0
    else:
        return 2000.0


def waste_heat_recovery_fraction() -> float:
    """
    Fraction of thermal load that can be covered by waste heat from main engine.
    Very simplified constant value.
    """
    return 0.5  # 50% of thermal demand covered by waste heat


# -----------------------------
# 3. Voyage simulation
# -----------------------------

def simulate_voyage(
    speed_knots: float,
    rated_power_kw: float = 15000.0,
    fuel_type: str = "HFO",
    scenario: str = "cruise",
    with_whr: bool = True,
) -> dict:
    """
    Simulate a voyage at constant speed.

    speed_knots: ship speed [knots]
    rated_power_kw: rated power of main engine [kW]
    fuel_type: "HFO", "LNG", "BIO"
    scenario: operation mode, affects hotel/thermal load
    with_whr: if True, waste heat recovery reduces boiler fuel

    returns: dict with fuel consumption and emissions
    """
    # Time to travel the distance
    travel_time_h = DISTANCE_NM / speed_knots

    # Propulsive power
    prop_power_kw = propulsive_power_kw(speed_knots)

    # Main engine fuel consumption
    fe_t_per_h = fuel_consumption_t_per_h(prop_power_kw, rated_power_kw)
    fe_total_t = fe_t_per_h * travel_time_h

    # Electrical load (assume covered by gensets burning same fuel type)
    el_kw = electrical_load_kw(scenario)
    # Assume genset efficiency: 220 g/kWh (demo)
    el_sfc_g_per_kwh = 220.0
    el_fuel_kg_per_h = el_sfc_g_per_kwh * el_kw / 1000.0
    el_fuel_t_per_h = el_fuel_kg_per_h / 1000.0
    el_total_t = el_fuel_t_per_h * travel_time_h

    # Thermal system & waste heat
    th_kw = thermal_load_kw(scenario)
    if with_whr:
        # Portion of thermal load covered by waste heat
        whr_frac = waste_heat_recovery_fraction()
    else:
        whr_frac = 0.0

    boiler_kw = th_kw * (1.0 - whr_frac)
    # Assume boiler efficiency: 90%, and energy/fuel gives a virtual SFC
    # We'll approximate fuel flow with a constant SFC as well (demo)
    boiler_sfc_g_per_kwh = 250.0
    boiler_fuel_kg_per_h = boiler_sfc_g_per_kwh * boiler_kw / 1000.0
    boiler_fuel_t_per_h = boiler_fuel_kg_per_h / 1000.0
    boiler_total_t = boiler_fuel_t_per_h * travel_time_h

    total_fuel_t = fe_total_t + el_total_t + boiler_total_t

    if fuel_type == "HFO":
        ef = EMISSION_FACTOR_HFO
    elif fuel_type == "LNG":
        ef = EMISSION_FACTOR_LNG
    else:
        ef = EMISSION_FACTOR_BIO

    total_co2_t = total_fuel_t * ef

    return {
        "speed_knots": speed_knots,
        "travel_time_h": travel_time_h,
        "fuel_main_t": fe_total_t,
        "fuel_electric_t": el_total_t,
        "fuel_boiler_t": boiler_total_t,
        "fuel_total_t": total_fuel_t,
        "co2_total_t": total_co2_t,
        "fuel_type": fuel_type,
        "with_whr": with_whr,
    }


# -----------------------------
# 4. Speed optimisation
# -----------------------------

def optimise_speed(
    fuel_type: str = "HFO",
    with_whr: bool = True,
    n_points: int = 200,
) -> dict:
    """
    Brute-force speed optimisation over a range, subject to arrival time constraint.

    Minimises total fuel while ensuring travel_time_h <= TIME_WINDOW_H.

    returns: dict with optimal result
    """
    speeds = np.linspace(MIN_SPEED_KTS, MAX_SPEED_KTS, n_points)
    results = []

    for v in speeds:
        res = simulate_voyage(
            speed_knots=v,
            fuel_type=fuel_type,
            with_whr=with_whr,
        )
        if res["travel_time_h"] <= TIME_WINDOW_H:
            results.append(res)

    if not results:
        raise ValueError("No feasible speed satisfies the time window.")

    df = pd.DataFrame(results)
    idx_min = df["fuel_total_t"].idxmin()
    best = df.loc[idx_min].to_dict()
    return best


# -----------------------------
# 5. Scenario comparison & plotting
# -----------------------------

def generate_speed_sweep_dataframe():
    """
    Generate a dataframe of fuel and CO2 vs speed for different decarbonisation scenarios.
    """
    speeds = np.linspace(MIN_SPEED_KTS, MAX_SPEED_KTS, 30)
    rows = []

    scenarios = [
        ("HFO", False, "Baseline HFO (no WHR)"),
        ("HFO", True, "HFO + Waste Heat Recovery"),
        ("LNG", True, "LNG + WHR"),
        ("BIO", True, "Biofuel + WHR"),
    ]

    for fuel_type, whr, label in scenarios:
        for v in speeds:
            res = simulate_voyage(
                speed_knots=v,
                fuel_type=fuel_type,
                with_whr=whr,
            )
            res["scenario_label"] = label
            rows.append(res)

    df = pd.DataFrame(rows)
    return df


def plot_results(df: pd.DataFrame):
    """
    Plot total fuel and CO2 vs speed for each scenario.
    """
    plt.figure()
    for label, sub in df.groupby("scenario_label"):
        plt.plot(sub["speed_knots"], sub["fuel_total_t"], marker="o", label=label)
    plt.xlabel("Speed [knots]")
    plt.ylabel("Total fuel for voyage [t]")
    plt.title("Fuel consumption vs speed for different scenarios")
    plt.grid(True)
    plt.legend()

    plt.figure()
    for label, sub in df.groupby("scenario_label"):
        plt.plot(sub["speed_knots"], sub["co2_total_t"], marker="o", label=label)
    plt.xlabel("Speed [knots]")
    plt.ylabel("Total CO2 emissions [t]")
    plt.title("CO2 emissions vs speed for different scenarios")
    plt.grid(True)
    plt.legend()

    plt.show()


# -----------------------------
# 6. Main entry point
# -----------------------------

def main():
    print("=== Ship Energy System Simulation & Optimization (Demo) ===")
    print(f"Voyage distance: {DISTANCE_NM} nm")
    print(f"Time window: {TIME_WINDOW_H} h")
    print(f"Speed range: {MIN_SPEED_KTS}–{MAX_SPEED_KTS} kn\n")

    # 1) Optimise speed for baseline HFO without WHR
    best_baseline = optimise_speed(fuel_type="HFO", with_whr=False)
    print("Optimal speed (Baseline HFO, no WHR):")
    print(f"  Speed:       {best_baseline['speed_knots']:.2f} kn")
    print(f"  Travel time: {best_baseline['travel_time_h']:.1f} h")
    print(f"  Fuel total:  {best_baseline['fuel_total_t']:.2f} t")
    print(f"  CO2 total:   {best_baseline['co2_total_t']:.2f} t\n")

    # 2) Optimise speed for HFO + WHR
    best_hfo_whr = optimise_speed(fuel_type="HFO", with_whr=True)
    print("Optimal speed (HFO + Waste Heat Recovery):")
    print(f"  Speed:       {best_hfo_whr['speed_knots']:.2f} kn")
    print(f"  Travel time: {best_hfo_whr['travel_time_h']:.1f} h")
    print(f"  Fuel total:  {best_hfo_whr['fuel_total_t']:.2f} t")
    print(f"  CO2 total:   {best_hfo_whr['co2_total_t']:.2f} t\n")

    # 3) Optimise speed for LNG + WHR
    best_lng = optimise_speed(fuel_type="LNG", with_whr=True)
    print("Optimal speed (LNG + WHR):")
    print(f"  Speed:       {best_lng['speed_knots']:.2f} kn")
    print(f"  Travel time: {best_lng['travel_time_h']:.1f} h")
    print(f"  Fuel total:  {best_lng['fuel_total_t']:.2f} t")
    print(f"  CO2 total:   {best_lng['co2_total_t']:.2f} t\n")

    # 4) Generate speed sweep and plot
    df = generate_speed_sweep_dataframe()
    print("Sample of scenario results:")
    print(df.head())

    plot_results(df)


if __name__ == "__main__":
    main()
