#!/usr/bin/env python3
# data_simulation.py
"""
Generate a synthetic aircraft fuel-consumption dataset and save train/test CSV files.

Features:
 - speed (m/s)
 - altitude (m)
 - payload (kg)
 - temperature (°C)
 - wind_speed (m/s)
 - air_density (kg/m^3)  <- computed from altitude & temperature (approximate ISA)

Target:
 - fuel_consumption (kg/hour) computed from a physics-inspired non-linear model
   (drag-based power estimate converted to fuel flow) + noise.

Creates:
 - data/train.csv
 - data/test.csv

Usage:
    python data_simulation.py
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple

RNG_SEED = 42


def compute_air_density(altitude_m: np.ndarray, temperature_c: np.ndarray) -> np.ndarray:
    """
    Approximate air density using an exponential decrease with altitude and an
    ideal-gas temperature correction. This is a simplification of ISA.

    rho = rho0 * exp(-altitude / scale_height) * (T0 / T)

    Args:
        altitude_m: altitude in meters
        temperature_c: temperature in Celsius

    Returns:
        air density in kg/m^3
    """
    rho0 = 1.225  # sea-level density kg/m^3
    scale_height = 8500.0  # meters, approx.
    # Convert temperature to Kelvin (avoid <=0 K)
    temp_k = temperature_c + 273.15
    temp_k = np.clip(temp_k, 200.0, None)
    rho = rho0 * np.exp(-altitude_m / scale_height) * (288.15 / temp_k)
    return rho


def simulate_fuel_consumption(
    speed: np.ndarray,
    altitude: np.ndarray,
    payload: np.ndarray,
    temperature: np.ndarray,
    wind_speed: np.ndarray,
    air_density: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute fuel consumption (kg/hour) using a simplified aerodynamics + propulsion model.

    Model intuition:
      - Parasite drag ~ 0.5 * rho * Cd0 * S * V^2
      - Induced drag ~ k * (W^2) / (0.5 * rho * S * V^2)
      - Drag force (N) -> Power_required = Drag * V (Watts)
      - Fuel mass flow (kg/s) = Power_required / (propulsive_efficiency * specific_energy)
      - Convert kg/s to kg/hour and add contributions from wind, payload inefficiencies, and noise.

    Args:
        speed: true airspeed (m/s)
        altitude: altitude (m)
        payload: payload mass (kg)
        temperature: ambient temperature (C)
        wind_speed: wind speed magnitude (m/s)
        air_density: air density (kg/m^3)
        rng: numpy random Generator for reproducibility

    Returns:
        fuel consumption in kg/hour
    """
    # Aircraft / physics constants (toy values chosen for realism-ish behavior)
    S = 30.0  # wing area m^2
    Cd0 = 0.025  # zero-lift drag coefficient
    k = 0.045  # induced drag factor
    g = 9.81  # m/s^2
    empty_weight = 7000.0  # kg (airframe + fuel-free base)
    propulsive_efficiency = 0.38  # overall propulsive/thermal efficiency (dimensionless)
    specific_energy = 43e6  # J/kg (jet fuel calorific value ~ 43 MJ/kg)

    # Weight (N)
    weight_N = (empty_weight + payload) * g

    # Avoid tiny speeds
    speed_safe = np.clip(speed, 20.0, None)

    # Parasite drag (N)
    parasite_drag = 0.5 * air_density * Cd0 * S * speed_safe ** 2

    # Induced drag (N) approximate
    induced_drag = k * (weight_N ** 2) / (0.5 * air_density * S * speed_safe ** 2 + 1e-6)

    # Total aerodynamic drag (N)
    total_drag = parasite_drag + induced_drag

    # Power required (W)
    power_required = total_drag * speed_safe

    # Fuel mass flow (kg/s)
    fuel_flow_kg_s = power_required / (propulsive_efficiency * specific_energy + 1e-9)

    # Wind penalty: more headwinds/crosswinds cause higher fuel burn (simplified)
    wind_penalty = 0.02 * wind_speed * (1.0 + payload / 10000.0)

    # Temperature penalty: higher temperature reduces air density -> slightly higher power in some regimes
    temp_penalty = 0.0008 * np.maximum(temperature - 10.0, 0.0)

    # Combine and convert to kg/hour
    fuel_flow_kg_hr = (fuel_flow_kg_s * (1.0 + wind_penalty + temp_penalty)) * 3600.0

    # Payload inefficiency: heavier payloads increase consumption beyond just weight in the drag term
    payload_effect = 0.00005 * payload * speed_safe / 100.0
    fuel_flow_kg_hr = fuel_flow_kg_hr + payload_effect

    # Add heteroscedastic noise (larger consumption -> slightly larger noise)
    noise_std = 0.02 * fuel_flow_kg_hr + 0.1
    noise = rng.normal(loc=0.0, scale=noise_std)

    fuel_flow_kg_hr_noisy = fuel_flow_kg_hr + noise

    # Ensure non-negative
    return np.clip(fuel_flow_kg_hr_noisy, 0.01, None)


def generate_dataset(n_samples: int = 10000, seed: int = RNG_SEED) -> pd.DataFrame:
    """
    Generate a pandas DataFrame containing features and target.

    Args:
        n_samples: number of rows to simulate
        seed: RNG seed for reproducibility

    Returns:
        DataFrame with columns: speed, altitude, payload, temperature, wind_speed, air_density, fuel_consumption
    """
    rng = np.random.default_rng(seed)

    # Sample realistic ranges
    # Speed: 70 - 250 m/s (approx 250 - 900 km/h)
    speed = rng.uniform(70.0, 250.0, size=n_samples)

    # Altitude: 500 - 12000 m
    altitude = rng.uniform(500.0, 12000.0, size=n_samples)

    # Payload: 0 - 20000 kg (small commuter -> medium transport)
    payload = rng.uniform(0.0, 20000.0, size=n_samples)

    # Temperature: -50°C (high altitude cold) to +30°C (hot day)
    temperature = rng.uniform(-50.0, 30.0, size=n_samples)

    # Wind speed: 0 - 60 m/s
    wind_speed = rng.uniform(0.0, 60.0, size=n_samples)

    # Compute air density from altitude and temperature
    air_density = compute_air_density(altitude, temperature)

    # Compute fuel consumption target
    fuel_consumption = simulate_fuel_consumption(
        speed=speed,
        altitude=altitude,
        payload=payload,
        temperature=temperature,
        wind_speed=wind_speed,
        air_density=air_density,
        rng=rng,
    )

    df = pd.DataFrame(
        {
            "speed": speed,
            "altitude": altitude,
            "payload": payload,
            "temperature": temperature,
            "wind_speed": wind_speed,
            "air_density": air_density,
            "fuel_consumption": fuel_consumption,
        }
    )

    return df


def save_train_test_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = RNG_SEED) -> Tuple[str, str]:
    """
    Split DataFrame into train/test and save CSV files under data/ directory.

    Returns:
        Tuple of (train_path, test_path)
    """
    from sklearn.model_selection import train_test_split

    os.makedirs("data", exist_ok=True)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)

    train_path = os.path.join("data", "train.csv")
    test_path = os.path.join("data", "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Also save metadata
    meta = {
        "n_total": len(df),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "seed": seed,
        "columns": df.columns.tolist(),
    }
    with open(os.path.join("data", "metadata.json"), "w", encoding="utf8") as fh:
        json.dump(meta, fh, indent=2)

    return train_path, test_path


def main():
    """Main entrypoint for dataset generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic aircraft fuel-consumption dataset.")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of samples to generate.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of samples to set aside for testing.")
    parser.add_argument("--seed", type=int, default=RNG_SEED, help="Random seed for reproducibility.")
    args = parser.parse_args()

    print(f"[INFO] Generating dataset with {args.n_samples} samples (seed={args.seed})...")
    df = generate_dataset(n_samples=args.n_samples, seed=args.seed)
    train_path, test_path = save_train_test_split(df, test_size=args.test_size, seed=args.seed)
    print(f"[INFO] Saved train -> {train_path}")
    print(f"[INFO] Saved test  -> {test_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
