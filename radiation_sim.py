

import numpy as np
import matplotlib.pyplot as plt

from environment_data import radiation




DAYS_TOTAL = radiation["days_in_model_month"]

MONTHLY_DOSE_RAD = radiation["monthly_dose_rad"]

STORM_FACTOR = radiation["reference_storm_factor"]




STORM_DURATION_DAYS = 1.0

STORM_START = (
    radiation["storm_day"]
    - STORM_DURATION_DAYS / 2.0
)

STORM_END = (
    radiation["storm_day"]
    + STORM_DURATION_DAYS / 2.0
)




N_POINTS = 3001

t_rad = np.linspace(
    0.0,
    DAYS_TOTAL,
    N_POINTS,
)

dt_rad = (
    t_rad[1]
    - t_rad[0]
)





quiet_dose_rate = (
    MONTHLY_DOSE_RAD
    /
    (
        DAYS_TOTAL
        + (STORM_FACTOR - 1.0)
        * STORM_DURATION_DAYS
    )
)

storm_dose_rate = (
    STORM_FACTOR
    * quiet_dose_rate
)



dose_rate = np.full_like(
    t_rad,
    quiet_dose_rate,
    dtype=float,
)

storm_mask = (
    (t_rad >= STORM_START)
    &
    (t_rad < STORM_END)
)

dose_rate[storm_mask] = (
    storm_dose_rate
)


# hello sir pls stop for identifcation !!

increments = (
    0.5
    * (
        dose_rate[1:]
        + dose_rate[:-1]
    )
    * np.diff(t_rad)
)

D_rad = np.concatenate(
    (
        [0.0],
        np.cumsum(increments),
    )
)




scale = (
    MONTHLY_DOSE_RAD
    / D_rad[-1]
)

dose_rate *= scale
D_rad *= scale

quiet_dose_rate *= scale
storm_dose_rate *= scale




final_dose = D_rad[-1]

calculated_ratio = (
    storm_dose_rate
    / quiet_dose_rate
)


#dododododoodododododod

if __name__ == "__main__":

    print(
        "HEO-1 Radiation Reference Case"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Reference month: "
        f"{radiation['month_name']} 2000"
    )

    print(
        f"Shielding: "
        f"{radiation['shielding_mil_be']:.0f} mil Be "
        f"(~{radiation['shielding_mil_al_equivalent']:.0f} "
        f"mil Al equivalent)"
    )

    print(
        f"Published monthly dose: "
        f"{MONTHLY_DOSE_RAD:.2f} Rad"
    )

    print(
        f"Integrated model dose: "
        f"{final_dose:.2f} Rad"
    )

    print(
        f"Quiet dose rate: "
        f"{quiet_dose_rate:.4f} Rad/day"
    )

    print(
        f"Storm dose rate: "
        f"{storm_dose_rate:.4f} Rad/day"
    )

    print(
        f"Storm / quiet ratio: "
        f"{calculated_ratio:.1f}x"
    )




    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        t_rad,
        dose_rate,
        label="Dose Rate",
    )

    plt.axvspan(
        STORM_START,
        STORM_END,
        alpha=0.2,
        label="10 Nov Storm Window",
    )

    plt.xlabel(
        "Time in November Model (days)"
    )

    plt.ylabel(
        "Dose Rate (Rad/day)"
    )

    plt.title(
        "HEO-1 Radiation Dose Rate "
        "(Open File 7389 Reference Case)"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()




    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        t_rad,
        D_rad,
        label="Cumulative Dose",
    )

    plt.axvspan(
        STORM_START,
        STORM_END,
        alpha=0.2,
        label="10 Nov Storm Window",
    )

    plt.axhline(
        MONTHLY_DOSE_RAD,
        linestyle="--",
        label=(
            f"Published November Dose "
            f"= {MONTHLY_DOSE_RAD:.2f} Rad"
        ),
    )

    plt.xlabel(
        "Time in November Model (days)"
    )

    plt.ylabel(
        "Cumulative Dose (Rad)"
    )

    plt.title(
        "HEO-1 Cumulative Radiation Dose "
        "(November 2000)"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()