
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import csv




from environment_data import (
    orbit,
    radiation,
    thermal,
    ceres_uncertainty,
)

from thermal_sim import (
    t_th,
    T_th,
    Q_sun,
    Q_IR,
    Q_albedo,
)

from radiation_sim import (
    t_rad,
    D_rad,
    dose_rate,
    STORM_START,
    STORM_END,
    MONTHLY_DOSE_RAD,
)

from degradation_model import (
    t_deg,
    T_deg,
    damage_deg,
    thermal_damage,
    radiation_damage,
    W_THERMAL,
    W_RADIATION,
    EA_J_MOL,
    T_REF_K,
)




OUTPUT_DIR = Path("outputs")

SUMMARY_DIR = OUTPUT_DIR / "summary"

SUMMARY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


NOMINAL_FILE = (
    OUTPUT_DIR
    / "model_predictions.npz"
)

MONTE_CARLO_FILE = (
    OUTPUT_DIR
    / "monte_carlo_results.npz"
)


if not NOMINAL_FILE.exists():

    raise FileNotFoundError(
        "\noutputs/model_predictions.npz was not found.\n"
        "Run train_models.py first."
    )


if not MONTE_CARLO_FILE.exists():

    raise FileNotFoundError(
        "\noutputs/monte_carlo_results.npz was not found.\n"
        "Run monte_carlo.py first."
    )




nominal = np.load(
    NOMINAL_FILE
)

mc = np.load(
    MONTE_CARLO_FILE
)




time = nominal["time"]

true_damage = nominal[
    "true_damage"
]

pred_time = nominal[
    "time_nn"
]

pred_environment = nominal[
    "environment_nn"
]

pred_pinn = nominal[
    "pinn"
]

train_indices = nominal[
    "train_indices"
]

test_indices = nominal[
    "test_indices"
]


time_rmse = float(
    nominal["time_rmse"]
)

environment_rmse = float(
    nominal["environment_rmse"]
)

pinn_rmse = float(
    nominal["pinn_rmse"]
)


time_mae = float(
    nominal["time_mae"]
)

environment_mae = float(
    nominal["environment_mae"]
)

pinn_mae = float(
    nominal["pinn_mae"]
)



storm_factors = mc[
    "storm_factor"
]

mc_rmse_time = mc[
    "rmse_time"
]

mc_rmse_environment = mc[
    "rmse_environment"
]

mc_rmse_pinn = mc[
    "rmse_pinn"
]


mc_rul_error_time = mc[
    "rul_error_time"
]

mc_rul_error_environment = mc[
    "rul_error_environment"
]

mc_rul_error_pinn = mc[
    "rul_error_pinn"
]


mc_time_rul = mc[
    "time_rul"
]

mc_environment_rul = mc[
    "environment_rul"
]

mc_pinn_rul = mc[
    "pinn_rul"
]


MC_CURRENT_TIME = float(
    mc["current_time_days"]
)

MC_FAILURE_THRESHOLD = float(
    mc["failure_threshold"]
)

MC_MEASUREMENT_NOISE = float(
    mc["measurement_noise_std"]
)


N_MONTE_CARLO = len(
    storm_factors
)




NOMINAL_FAILURE_THRESHOLD = 0.85

NOMINAL_CURRENT_TIME = 0.0



FIGURE_DPI = 300



SHOW_FIGURES = True




def finish_figure(
    filename,
):

    plt.tight_layout()

    plt.savefig(
        SUMMARY_DIR / filename,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    if SHOW_FIGURES:

        plt.show()

    else:

        plt.close()




def find_threshold_crossing(
    time_array,
    damage_array,
    threshold,
):

    time_array = np.asarray(
        time_array
    ).flatten()

    damage_array = np.asarray(
        damage_array
    ).flatten()


    indices = np.where(
        damage_array >= threshold
    )[0]


    if len(indices) == 0:

        return np.nan


    upper = indices[0]


    if upper == 0:

        return float(
            time_array[0]
        )


    lower = upper - 1


    t1 = time_array[lower]
    t2 = time_array[upper]

    d1 = damage_array[lower]
    d2 = damage_array[upper]


    if np.isclose(
        d1,
        d2,
    ):

        return float(t2)


    return float(

        t1

        +

        (
            threshold
            - d1
        )

        /

        (
            d2
            - d1
        )

        *

        (
            t2
            - t1
        )
    )




plt.figure(
    figsize=(9, 5)
)


plt.plot(
    t_th / 3600.0,
    Q_sun,
    label="Direct Solar Flux",
)

plt.plot(
    t_th / 3600.0,
    Q_IR,
    label="Earth IR Flux",
)

plt.plot(
    t_th / 3600.0,
    Q_albedo,
    label="Albedo Flux",
)


plt.xlabel(
    "Time (hours)"
)

plt.ylabel(
    "Incident Flux (W/m²)"
)

plt.title(
    "HEO Thermal Environment "
    "(CERES-Informed Model)"
)

plt.grid(True)

plt.legend()


finish_figure(
    "01_thermal_environment.png"
)




plt.figure(
    figsize=(9, 5)
)


plt.plot(
    t_th / 3600.0,
    T_th,
    linewidth=2,
)


plt.xlabel(
    "Time (hours)"
)

plt.ylabel(
    "Temperature (K)"
)

plt.title(
    "HEO Temperature Over One Orbit "
    "(CERES-Informed Model)"
)

plt.grid(True)


finish_figure(
    "02_temperature_profile.png"
)




plt.figure(
    figsize=(9, 5)
)


plt.plot(
    t_rad,
    D_rad,
    linewidth=2,
    label="Cumulative Dose",
)


plt.axvspan(
    STORM_START,
    STORM_END,
    alpha=0.15,
    label="10 Nov Storm Window",
)


plt.axhline(
    MONTHLY_DOSE_RAD,
    linestyle="--",
    label=(
        f"November HEO-1 Dose "
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


finish_figure(
    "03_radiation_dose.png"
)





plt.figure(
    figsize=(9, 5)
)


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
    STORM_START,
    STORM_END,
    alpha=0.15,
    label="Radiation Storm",
)


plt.xlabel(
    "Time (days)"
)

plt.ylabel(
    "Normalized Degradation / Stress Index"
)

plt.title(
    "Combined DC-DC Converter Stress Model "
    "(HEO Environment)"
)

plt.grid(True)

plt.legend()


finish_figure(
    "04_combined_degradation.png"
)




plt.figure(
    figsize=(10, 6)
)


plt.plot(
    time,
    true_damage,
    linewidth=2.5,
    label="Reference Degradation",
)


plt.scatter(
    time[train_indices],
    true_damage[train_indices],
    s=15,
    label="Training Observations",
)


plt.plot(
    time,
    pred_time,
    "--",
    label="Time-Only NN",
)


plt.plot(
    time,
    pred_environment,
    "--",
    label="Environmental NN",
)


plt.plot(
    time,
    pred_pinn,
    linewidth=2,
    label="Physics PINN",
)


plt.xlabel(
    "Time (days)"
)

plt.ylabel(
    "Normalized Degradation Index"
)

plt.title(
    "Nominal Degradation Prediction Comparison"
)

plt.grid(True)

plt.legend()


finish_figure(
    "05_nominal_model_comparison.png"
)




models = [
    "Time-Only NN",
    "Environmental NN",
    "Physics PINN",
]


nominal_rmse = [
    time_rmse,
    environment_rmse,
    pinn_rmse,
]


nominal_mae = [
    time_mae,
    environment_mae,
    pinn_mae,
]


x = np.arange(
    len(models)
)

width = 0.36


plt.figure(
    figsize=(9, 5)
)


plt.bar(
    x - width / 2.0,
    nominal_rmse,
    width,
    label="RMSE",
)

plt.bar(
    x + width / 2.0,
    nominal_mae,
    width,
    label="MAE",
)


plt.xticks(
    x,
    models,
)

plt.ylabel(
    "Prediction Error"
)

plt.title(
    "Nominal Held-Out Prediction Error"
)

plt.grid(
    True,
    axis="y",
)

plt.legend()


finish_figure(
    "06_nominal_error_comparison.png"
)




plt.figure(
    figsize=(9, 5)
)


plt.boxplot(
    [
        mc_rmse_time,
        mc_rmse_environment,
        mc_rmse_pinn,
    ],

    tick_labels=[
        "Time-Only NN",
        "Environmental NN",
        "Physics PINN",
    ],
)


plt.ylabel(
    "Future Degradation RMSE"
)

plt.title(
    "Monte Carlo Future-Prediction Accuracy "
    "(Pre-Storm Observations)"
)

plt.grid(
    True,
    axis="y",
)


finish_figure(
    "07_monte_carlo_future_rmse.png"
)




plt.figure(
    figsize=(9, 5)
)


plt.scatter(
    storm_factors,
    mc_rmse_environment,
    label="Environmental NN",
)

plt.scatter(
    storm_factors,
    mc_rmse_pinn,
    label="Physics PINN",
)


plt.xlabel(
    "Storm Enhancement Factor"
)

plt.ylabel(
    "Future Degradation RMSE"
)

plt.title(
    "Prediction Error vs HEO Storm Severity"
)

plt.grid(True)

plt.legend()


finish_figure(
    "08_storm_severity_vs_error.png"
)




nominal_failure_time = (
    find_threshold_crossing(

        time,

        pred_pinn,

        NOMINAL_FAILURE_THRESHOLD,
    )
)


reference_failure_time = (
    find_threshold_crossing(

        time,

        true_damage,

        NOMINAL_FAILURE_THRESHOLD,
    )
)


nominal_rul = (

    nominal_failure_time
    - NOMINAL_CURRENT_TIME

    if np.isfinite(
        nominal_failure_time
    )

    else np.nan
)


plt.figure(
    figsize=(9, 5)
)


plt.plot(
    time,
    pred_pinn,
    linewidth=2,
    label="PINN Predicted Degradation",
)


plt.plot(
    time,
    true_damage,
    "--",
    label="Reference Degradation",
)


plt.axhline(
    NOMINAL_FAILURE_THRESHOLD,
    linestyle="--",
    label=(
        "Assumed Failure Threshold "
        f"= {NOMINAL_FAILURE_THRESHOLD:.2f}"
    ),
)


if np.isfinite(
    nominal_failure_time
):

    plt.axvline(
        nominal_failure_time,
        linestyle=":",
        label=(
            f"PINN Failure Time "
            f"= {nominal_failure_time:.2f} d"
        ),
    )


    plt.scatter(
        [nominal_failure_time],
        [NOMINAL_FAILURE_THRESHOLD],
        s=70,
    )


plt.xlabel(
    "Time (days)"
)

plt.ylabel(
    "Normalized Degradation Index"
)

plt.title(
    "Nominal Remaining Useful Life Estimation "
    "Using Physics PINN"
)

plt.grid(True)

plt.legend()


finish_figure(
    "09_nominal_rul.png"
)




valid_time = int(
    np.sum(
        np.isfinite(
            mc_time_rul
        )
    )
)

valid_environment = int(
    np.sum(
        np.isfinite(
            mc_environment_rul
        )
    )
)

valid_pinn = int(
    np.sum(
        np.isfinite(
            mc_pinn_rul
        )
    )
)


valid_counts = [
    valid_time,
    valid_environment,
    valid_pinn,
]


valid_percent = (

    100.0
    * np.asarray(
        valid_counts
    )
    / N_MONTE_CARLO
)


plt.figure(
    figsize=(9, 5)
)


bars = plt.bar(
    models,
    valid_percent,
)


plt.ylabel(
    "Valid Future Threshold Predictions (%)"
)

plt.title(
    "Monte Carlo Future-Failure Detection Rate"
)

plt.ylim(
    0,
    100,
)

plt.grid(
    True,
    axis="y",
)


for bar, percentage, count in zip(
    bars,
    valid_percent,
    valid_counts,
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2.0,

        percentage + 2,

        f"{count}/{N_MONTE_CARLO}",

        ha="center",
    )


finish_figure(
    "10_rul_prediction_success_rate.png"
)




def finite_stats(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]


    if len(values) == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
        )


    return (

        float(
            np.mean(values)
        ),

        float(
            np.std(values)
        ),

        float(
            np.median(values)
        ),
    )


mc_time_stats = finite_stats(
    mc_rmse_time
)

mc_environment_stats = finite_stats(
    mc_rmse_environment
)

mc_pinn_stats = finite_stats(
    mc_rmse_pinn
)


rul_time_stats = finite_stats(
    mc_rul_error_time
)

rul_environment_stats = finite_stats(
    mc_rul_error_environment
)

rul_pinn_stats = finite_stats(
    mc_rul_error_pinn
)


summary_rows = [

    [
        "Time-Only NN",
        time_rmse,
        time_mae,
        mc_time_stats[0],
        mc_time_stats[1],
        valid_time,
        rul_time_stats[0],
    ],

    [
        "Environmental NN",
        environment_rmse,
        environment_mae,
        mc_environment_stats[0],
        mc_environment_stats[1],
        valid_environment,
        rul_environment_stats[0],
    ],

    [
        "Physics PINN",
        pinn_rmse,
        pinn_mae,
        mc_pinn_stats[0],
        mc_pinn_stats[1],
        valid_pinn,
        rul_pinn_stats[0],
    ],
]


summary_csv = (
    SUMMARY_DIR
    / "model_performance_summary.csv"
)


with open(
    summary_csv,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(
        file
    )


    writer.writerow(
        [
            "Model",
            "Nominal RMSE",
            "Nominal MAE",
            "Monte Carlo Future RMSE Mean",
            "Monte Carlo Future RMSE Std",
            "Valid Future RUL Predictions",
            "Mean Absolute RUL Error When Valid [days]",
        ]
    )


    writer.writerows(
        summary_rows
    )





provenance_rows = [

    [
        "HEO orbital period",
        orbit["period_hours"],
        "hours",
        "Source-derived",
        "GSC Open File 7389 / HEO-3",
    ],

    [
        "Semi-major axis",
        orbit["semi_major_axis_km"],
        "km",
        "Source-derived",
        "GSC Open File 7389 example Molniya orbit",
    ],

    [
        "Orbital eccentricity",
        orbit["eccentricity"],
        "-",
        "Source-derived",
        "GSC Open File 7389 example Molniya orbit",
    ],

    [
        "Inclination",
        orbit["inclination_deg"],
        "deg",
        "Source-derived",
        "GSC Open File 7389",
    ],

    [
        "November HEO-1 monthly dose",
        radiation["monthly_dose_rad"],
        "Rad",
        "Source-derived",
        "GSC Open File 7389, HEO-1 Dosimeter 1",
    ],

    [
        "Radiation shielding",
        radiation["shielding_mil_be"],
        "mil Be",
        "Source-derived",
        "GSC Open File 7389",
    ],

    [
        "Storm enhancement range",
        (
            f"{radiation['storm_factor_min']}"
            f"-{radiation['storm_factor_max']}"
        ),
        "x quiet-day level",
        "Source-derived",
        "GSC Open File 7389",
    ],

    [
        "Reference storm enhancement",
        radiation[
            "reference_storm_factor"
        ],
        "x",
        "Modeling choice based on source lower bound",
        "Reference deterministic scenario",
    ],

    [
        "Modeled storm duration",
        1.0,
        "day",
        "Modeling assumption",
        "Storm-day representation",
    ],

    [
        "Solar constant",
        thermal[
            "solar_constant_W_m2"
        ],
        "W/m^2",
        "Source-derived",
        "CERES-based Sasaki thermal paper",
    ],

    [
        "Surface absorptivity",
        thermal[
            "absorptivity"
        ],
        "-",
        "Source-derived model parameter",
        "Sasaki single-node demonstration",
    ],

    [
        "Surface emissivity",
        thermal[
            "emissivity"
        ],
        "-",
        "Source-derived model parameter",
        "Sasaki single-node demonstration",
    ],

    [
        "High heat-capacity parameter",
        thermal[
            "node_heat_capacity_J_K"
        ],
        "unit-area thermal-capacity parameter",
        "Source-derived model case",
        "Sasaki high-capacity single-node case",
    ],

    [
        "CERES OLR model RMSE",
        ceres_uncertainty[
            "OLR_RMSE_W_m2"
        ],
        "W/m^2",
        "Source-derived",
        "Sasaki CERES model",
    ],

    [
        "CERES albedo model RMSE",
        ceres_uncertainty[
            "albedo_RMSE"
        ],
        "-",
        "Source-derived",
        "Sasaki CERES model",
    ],

    [
        "Arrhenius activation energy",
        EA_J_MOL,
        "J/mol",
        "Modeling assumption",
        "Not supplied by environmental PDFs",
    ],

    [
        "Arrhenius reference temperature",
        T_REF_K,
        "K",
        "Modeling assumption",
        "Not supplied by environmental PDFs",
    ],

    [
        "Thermal degradation weight",
        W_THERMAL,
        "-",
        "Modeling assumption",
        "Equal-weight normalized stress model",
    ],

    [
        "Radiation degradation weight",
        W_RADIATION,
        "-",
        "Modeling assumption",
        "Equal-weight normalized stress model",
    ],

    [
        "Failure threshold",
        NOMINAL_FAILURE_THRESHOLD,
        "normalized degradation",
        "Modeling assumption",
        "Demonstration RUL criterion",
    ],

    [
        "Monte Carlo current time",
        MC_CURRENT_TIME,
        "days",
        "Validation design choice",
        "Pre-storm prediction scenario",
    ],

    [
        "Monte Carlo measurement noise",
        MC_MEASUREMENT_NOISE,
        "normalized degradation std",
        "Modeling assumption",
        "Synthetic observation uncertainty",
    ],

    [
        "Monte Carlo storm distribution",
        "Uniform",
        "10-100x",
        "Modeling assumption",
        "Source gives range, not probability distribution",
    ],
]


provenance_csv = (
    SUMMARY_DIR
    / "data_provenance.csv"
)


with open(
    provenance_csv,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(
        file
    )


    writer.writerow(
        [
            "Parameter",
            "Value",
            "Unit",
            "Classification",
            "Basis",
        ]
    )


    writer.writerows(
        provenance_rows
    )




monte_carlo_csv = (
    SUMMARY_DIR
    / "monte_carlo_summary.csv"
)


with open(
    monte_carlo_csv,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(
        file
    )


    writer.writerow(
        [
            "Model",
            "Future RMSE Mean",
            "Future RMSE Std",
            "Future RMSE Median",
            "Valid RUL Predictions",
            "Total Runs",
            "Valid Prediction Rate [%]",
            "Mean Absolute RUL Error When Valid [days]",
        ]
    )


    writer.writerow(
        [
            "Time-Only NN",

            mc_time_stats[0],
            mc_time_stats[1],
            mc_time_stats[2],

            valid_time,
            N_MONTE_CARLO,

            100.0
            * valid_time
            / N_MONTE_CARLO,

            rul_time_stats[0],
        ]
    )


    writer.writerow(
        [
            "Environmental NN",

            mc_environment_stats[0],
            mc_environment_stats[1],
            mc_environment_stats[2],

            valid_environment,
            N_MONTE_CARLO,

            100.0
            * valid_environment
            / N_MONTE_CARLO,

            rul_environment_stats[0],
        ]
    )


    writer.writerow(
        [
            "Physics PINN",

            mc_pinn_stats[0],
            mc_pinn_stats[1],
            mc_pinn_stats[2],

            valid_pinn,
            N_MONTE_CARLO,

            100.0
            * valid_pinn
            / N_MONTE_CARLO,

            rul_pinn_stats[0],
        ]
    )




print(
    "\n"
    "======================================================"
)

print(
    "FINAL PROJECT RESULTS SUMMARY"
)

print(
    "======================================================"
)


print(
    "\nNOMINAL HELD-OUT PREDICTION"
)

print(
    "------------------------------------------------------"
)

print(
    f"Time-only NN      RMSE = {time_rmse:.6f}"
)

print(
    f"Environmental NN  RMSE = {environment_rmse:.6f}"
)

print(
    f"Physics PINN      RMSE = {pinn_rmse:.6f}"
)


print(
    "\nMONTE CARLO FUTURE-PREDICTION RMSE"
)

print(
    "------------------------------------------------------"
)

print(
    f"Time-only NN      "
    f"{mc_time_stats[0]:.6f} "
    f"+/- {mc_time_stats[1]:.6f}"
)

print(
    f"Environmental NN  "
    f"{mc_environment_stats[0]:.6f} "
    f"+/- {mc_environment_stats[1]:.6f}"
)

print(
    f"Physics PINN      "
    f"{mc_pinn_stats[0]:.6f} "
    f"+/- {mc_pinn_stats[1]:.6f}"
)


print(
    "\nVALID FUTURE FAILURE / RUL PREDICTIONS"
)

print(
    "------------------------------------------------------"
)

print(
    f"Time-only NN:      "
    f"{valid_time}/{N_MONTE_CARLO}"
)

print(
    f"Environmental NN:  "
    f"{valid_environment}/{N_MONTE_CARLO}"
)

print(
    f"Physics PINN:      "
    f"{valid_pinn}/{N_MONTE_CARLO}"
)


print(
    "\nNOMINAL PINN RUL DEMONSTRATION"
)

print(
    "------------------------------------------------------"
)


if np.isfinite(
    nominal_failure_time
):

    print(
        f"Assumed failure threshold = "
        f"{NOMINAL_FAILURE_THRESHOLD:.2f}"
    )

    print(
        f"PINN predicted failure time = "
        f"{nominal_failure_time:.3f} days"
    )

    print(
        f"Nominal RUL from t = 0 = "
        f"{nominal_rul:.3f} days"
    )

    print(
        f"Reference threshold crossing = "
        f"{reference_failure_time:.3f} days"
    )


print(
    "\nIMPORTANT INTERPRETATION"
)

print(
    "------------------------------------------------------"
)

print(
    "Environmental inputs are source-grounded."
)

print(
    "The degradation weights, Arrhenius converter parameters,"
)

print(
    "and normalized failure threshold are modeling assumptions."
)

print(
    "Therefore the nominal RUL demonstrates the methodology;"
)

print(
    "it is not an experimentally established lifetime for a"
)

print(
    "specific flight DC-DC converter."
)


print(
    "\nFiles saved to:"
)

print(
    SUMMARY_DIR.resolve()
)

print(
    "\nDone."
)