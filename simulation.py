"""
simulation.py

Core ship energy system simulation:
- propulsion
- main engine fuel consumption
- electrical & thermal loads
- waste heat recovery
- voyage simulation
- speed optimisation
"""

import numpy as np
import pandas as pd

# -----------------------------
# Constants & demo parameters
# -----------------------------

DISTANCE_NM_DEFAULT = 1000.0   # default voyage distance
TIME_WINDOW_H_DEFAULT = 60.0   # default max allowed time [h]

MIN_SPEED_KTS = 10.0
MAX_SPEED_KTS = 20.0

EMISSION_FACTOR_HFO = 3.114  # tCO2 / t fuel (demo)
EMISSION_FACTOR_LNG = 2.75   # tCO2 / t fuel (demo)
EMISSION_FACTOR_BIO = 0.3    # tCO2 / t fuel (demo)


# -----------------------------
# Basic models
# -----------------------------

def propulsive_power_kw(speed_knots: float) -> float:
    """Very simplified propulsion model: P ~ v^3."""
    speed_ms = speed_knots * 0.51444
    a = 0.8  # demo constant
    return a * speed_ms**3


def main_engine_sfc_g_per_kwh(power_fraction: float) -> float:
    """
    Specific fuel consumption curve (demo):
    best around 80% load, worse at low/high loads.
    """
    base = 170.0  # g/kWh
    penalty = 60.0 * (power_fraction - 0.8) ** 2
    return base + penalty


def fuel_consumption_t_per_h(power_kw: float, rated_power_kw: float) -> float:
    """Hourly fuel consumption of main engine in tons/hour."""
    if power_kw <= 0:
        return 0.0

    load_fraction = min(max(power_kw / rated_power_kw, 0.05), 1.0)
    sfc = main_engine_sfc_g_per_kwh(load_fraction)
    fuel_kg_per_h = sfc * power_kw / 1000.0
    return fuel_kg_per_h / 1000.0  # t/h


def electrical_load_kw(scenario: str = "cruise") -> float:
    """Hotel & service electrical load [kW]."""
    if scenario == "port":
        return 1500.0
    if scenario == "maneuvering":
        return 2500.0
    return 3000.0  # cruise


def thermal_load_kw(scenario: str = "cruise") -> float:
    """Thermal demand for heating etc. [kW]."""
    if scenario == "port":
        return 1000.0
    if scenario == "maneuvering":
        return 1500.0
    return 2000.0


def waste_heat_recovery_fraction() -> float:
    """Fraction of thermal load covered by waste heat."""
    return 0.5  # 50% (demo)


def emission_factor_for_fuel(fuel_type: str) -> float:
    fuel_type = fuel_type.upper()
    if fuel_type == "LNG":
        return EMISSION_FACTOR_LNG
    if fuel_type == "BIO":
        return EMISSION_FACTOR_BIO
    return EMISSION_FACTOR_HFO  # default HFO


# -----------------------------
# Voyage simulation
# -----------------------------

def simulate_voyage(
    speed_knots: float,
    distance_nm: float = DISTANCE_NM_DEFAULT,
    rated_power_kw: float = 15000.0,
    fuel_type: str = "HFO",
    scenario: str = "cruise",
    with_whr: bool = True,
) -> dict:
    """
    Simulate a voyage at constant speed.

    Returns dict with:
    - travel_time_h
    - fuel_main_t
    - fuel_electric_t
    - fuel_boiler_t
    - fuel_total_t
    - co2_total_t
    """
    travel_time_h = distance_nm / speed_knots

    # Propulsion
    prop_power_kw = propulsive_power_kw(speed_knots)
    fe_t_per_h = fuel_consumption_t_per_h(prop_power_kw, rated_power_kw)
    fe_total_t = fe_t_per_h * travel_time_h

    # Electrical (gensets)
    el_kw = electrical_load_kw(scenario)
    el_sfc_g_per_kwh = 220.0  # demo
    el_fuel_kg_per_h = el_sfc_g_per_kwh * el_kw / 1000.0
    el_fuel_t_per_h = el_fuel_kg_per_h / 1000.0
    el_total_t = el_fuel_t_per_h * travel_time_h

    # Thermal & WHR
    th_kw = thermal_load_kw(scenario)
    whr_frac = waste_heat_recovery_fraction() if with_whr else 0.0
    boiler_kw = th_kw * (1.0 - whr_frac)

    boiler_sfc_g_per_kwh = 250.0
    boiler_fuel_kg_per_h = boiler_sfc_g_per_kwh * boiler_kw / 1000.0
    boiler_fuel_t_per_h = boiler_fuel_kg_per_h / 1000.0
    boiler_total_t = boiler_fuel_t_per_h * travel_time_h

    fuel_total_t = fe_total_t + el_total_t + boiler_total_t
    ef = emission_factor_for_fuel(fuel_type)
    co2_total_t = fuel_total_t * ef

    return {
        "speed_knots": speed_knots,
        "distance_nm": distance_nm,
        "travel_time_h": travel_time_h,
        "fuel_main_t": fe_total_t,
        "fuel_electric_t": el_total_t,
        "fuel_boiler_t": boiler_total_t,
        "fuel_total_t": fuel_total_t,
        "co2_total_t": co2_total_t,
        "fuel_type": fuel_type.upper(),
        "with_whr": with_whr,
        "scenario": scenario,
    }


# -----------------------------
# Speed optimisation
# -----------------------------

def optimise_speed(
    distance_nm: float = DISTANCE_NM_DEFAULT,
    time_window_h: float = TIME_WINDOW_H_DEFAULT,
    fuel_type: str = "HFO",
    with_whr: bool = True,
    n_points: int = 200,
) -> dict:
    """
    Brute-force speed optimisation: minimise total fuel
    subject to travel_time_h <= time_window_h.
    """
    speeds = np.linspace(MIN_SPEED_KTS, MAX_SPEED_KTS, n_points)
    results = []

    for v in speeds:
        res = simulate_voyage(
            speed_knots=v,
            distance_nm=distance_nm,
            fuel_type=fuel_type,
            with_whr=with_whr,
        )
        if res["travel_time_h"] <= time_window_h:
            results.append(res)

    if not results:
        raise ValueError("No feasible speed satisfies the time window.")

    df = pd.DataFrame(results)
    idx_min = df["fuel_total_t"].idxmin()
    return df.loc[idx_min].to_dict()


def speed_sweep(
    distance_nm: float,
    fuel_type: str,
    with_whr: bool,
    n_points: int = 30,
) -> pd.DataFrame:
    """
    Sweep over speed range and return a DataFrame with results.
    """
    speeds = np.linspace(MIN_SPEED_KTS, MAX_SPEED_KTS, n_points)
    rows = []
    for v in speeds:
        res = simulate_voyage(
            speed_knots=v,
            distance_nm=distance_nm,
            fuel_type=fuel_type,
            with_whr=with_whr,
        )
        rows.append(res)
    return pd.DataFrame(rows)
