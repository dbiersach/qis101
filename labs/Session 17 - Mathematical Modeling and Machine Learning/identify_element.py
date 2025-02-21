#!/usr/bin/env python3
"""identify_element.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def fit_linear(x, y):
    m = len(x) * np.sum(x * y) - np.sum(x) * np.sum(y)
    m = m / (len(x) * np.sum(x**2) - np.sum(x) ** 2)
    b = (np.sum(y) - m * np.sum(x)) / len(x)
    return m, b


def main():
    # Read experiment data from data file
    file_path = Path(__file__).parent / "gas.csv"
    temperature, volume = np.genfromtxt(file_path, delimiter=",", unpack=True)

    # Convert experiment data to SI units
    temperature += 273.15  # celsius to kelvin
    volume /= 1000  # liters to meters^3

    # Calculate line of best fit
    slope, yint = fit_linear(temperature, volume)

    # Apply Ideal gas law
    p = 2.0 * 101_325  # Convert 2.0 atm (given) to pascals
    r = 8.31446261815324  # Gas constant (SI units)
    n = p / r * slope  # Moles of gas (rearrange ideal gas law equation)

    # Calculate atomic mass of sample using number of moles
    m_sample = 50  # (given) grams
    molar_mass = m_sample / n  # sample mass divided by number of moles

    # Create array of Kelvin temperatures from 0 to 500 inclusive
    t = np.linspace(0, 500)

    plt.figure(Path(__file__).name)
    plt.scatter(temperature, volume, color="red")
    plt.plot(t, slope * t + yint)
    plt.title(f"Unknown Gas ({molar_mass:.3f}u)")
    plt.xlabel(r"Temperature $(\degree K)$")
    plt.ylabel("Volume $(m^3)$")
    plt.grid("on")
    plt.show()


if __name__ == "__main__":
    main()
