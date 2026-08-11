

import numpy as np
import matplotlib.pyplot as plt

from thermal_sim import t_th, T_th
from radiation_sim import (
    t_rad,
    D_rad,
    dose_rate,
)


# do u know  where am i?


t_deg = t_rad.copy()


t_deg_s = t_deg * 24.0 * 3600.0




orbit_period_s = t_th[-1]


thermal_phase_s = np.mod(
    t_deg_s,
    orbit_period_s,
)


T_deg = np.interp(
    thermal_phase_s,
    t_th,
    T_th,
)




R_GAS = 8.314462618

EA_J_MOL = 70_000.0
T_REF_K = 300.0




thermal_acceleration = np.exp(
    (EA_J_MOL / R_GAS)
    * (
        1.0 / T_REF_K
        - 1.0 / T_deg
    )
)




thermal_increment = (
    0.5
    * (
        thermal_acceleration[1:]
        + thermal_acceleration[:-1]
    )
    * np.diff(t_deg)
)

thermal_exposure_days = np.concatenate(
    (
        [0.0],
        np.cumsum(thermal_increment),
    )
)




if thermal_exposure_days[-1] > 0:

    thermal_damage = (
        thermal_exposure_days
        / thermal_exposure_days[-1]
    )

else:

    thermal_damage = np.zeros_like(
        thermal_exposure_days
    )




radiation_reference_dose = D_rad[-1]

radiation_damage = (
    D_rad
    / radiation_reference_dose
)




W_THERMAL = 0.50
W_RADIATION = 0.50


damage_deg = (
    W_THERMAL * thermal_damage
    + W_RADIATION * radiation_damage
)



damage_deg = np.clip(
    damage_deg,
    0.0,
    None,
)




damage_rate = np.gradient(
    damage_deg,
    t_deg,
)




if __name__ == "__main__":

    print(
        "Combined Degradation Model"
    )

    print(
        "--------------------------"
    )

    print(
        f"Temperature range over repeated HEO cycle: "
        f"{T_deg.min():.2f} - {T_deg.max():.2f} K"
    )

    print(
        f"Equivalent thermal exposure after 30 days: "
        f"{thermal_exposure_days[-1]:.3f} "
        f"reference-temperature days"
    )

    print(
        f"Final normalized thermal stress: "
        f"{thermal_damage[-1]:.3f}"
    )

    print(
        f"Final normalized radiation stress: "
        f"{radiation_damage[-1]:.3f}"
    )

    print(
        f"Final combined degradation index: "
        f"{damage_deg[-1]:.3f}"
    )




    plt.figure(figsize=(9, 5))

    plt.plot(
        t_deg,
        damage_deg,
        linewidth=2,
        label="Combined Degradation Index",
    )

    plt.plot(
        t_deg,
        thermal_damage,
        "--",
        label="Thermal Stress Index",
    )

    plt.plot(
        t_deg,
        radiation_damage,
        "--",
        label="Radiation Stress Index",
    )

    plt.axvspan(
        9.5,
        10.5,
        alpha=0.15,
        label="10 Nov Radiation Storm",
    )

    plt.xlabel("Time (days)")
    plt.ylabel("Normalized Degradation / Stress Index")

    plt.title(
        "Combined DC-DC Converter Stress Model "
        "(HEO Environment)"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # u have the right to remain silent !

    first_three_days = (
        t_deg <= 3.0
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        t_deg[first_three_days],
        T_deg[first_three_days],
    )

    plt.xlabel("Time (days)")
    plt.ylabel("Temperature (K)")

    plt.title(
        "Repeated HEO Thermal Exposure "
        "(First Three Days)"
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()