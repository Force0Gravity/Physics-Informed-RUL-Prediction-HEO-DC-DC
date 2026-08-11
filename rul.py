

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt



RESULTS_FILE = Path(
    "outputs/model_predictions.npz"
)




FAILURE_THRESHOLD = 0.85




CURRENT_TIME_DAYS = 0.0




if not RESULTS_FILE.exists():

    raise FileNotFoundError(
        "\nModel prediction file was not found.\n"
        "Run train_models.py first.\n"
        f"Expected file:\n{RESULTS_FILE}"
    )


results = np.load(
    RESULTS_FILE
)


t = results[
    "time"
]

true_damage = results[
    "true_damage"
]

pinn_damage = results[
    "pinn"
]




def find_threshold_crossing(
    time,
    damage,
    threshold,
):


    damage = np.asarray(
        damage
    ).flatten()

    time = np.asarray(
        time
    ).flatten()



    crossing_indices = np.where(
        damage >= threshold
    )[0]


    if len(crossing_indices) == 0:

        return None


    upper_index = crossing_indices[0]



    if upper_index == 0:

        return float(
            time[0]
        )


    lower_index = (
        upper_index - 1
    )


    t1 = time[
        lower_index
    ]

    t2 = time[
        upper_index
    ]


    d1 = damage[
        lower_index
    ]

    d2 = damage[
        upper_index
    ]


    if np.isclose(
        d2,
        d1,
    ):

        return float(
            t2
        )




    failure_time = (

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


    return float(
        failure_time
    )




pinn_failure_time = (
    find_threshold_crossing(

        time=t,

        damage=pinn_damage,

        threshold=FAILURE_THRESHOLD,
    )
)




reference_failure_time = (
    find_threshold_crossing(

        time=t,

        damage=true_damage,

        threshold=FAILURE_THRESHOLD,
    )
)




if pinn_failure_time is None:

    pinn_rul = None

else:

    pinn_rul = (
        pinn_failure_time
        - CURRENT_TIME_DAYS
    )




print(
    "\n======================================"
)

print(
    "PINN REMAINING USEFUL LIFE ESTIMATION"
)

print(
    "======================================"
)


print(
    f"Assumed failure threshold: "
    f"{FAILURE_THRESHOLD:.2f}"
)


print(
    f"Current operating time: "
    f"{CURRENT_TIME_DAYS:.2f} days"
)


if pinn_failure_time is None:

    print(
        "\nPINN does not reach the selected "
        "failure threshold within the "
        "30-day prediction window."
    )

    print(
        f"Therefore RUL > "
        f"{t[-1] - CURRENT_TIME_DAYS:.2f} days."
    )


else:

    print(
        f"\nPINN predicted failure time: "
        f"{pinn_failure_time:.3f} days"
    )

    print(
        f"PINN Remaining Useful Life: "
        f"{pinn_rul:.3f} days"
    )


if reference_failure_time is not None:

    print(
        f"\nReference-model failure time: "
        f"{reference_failure_time:.3f} days"
    )


    if pinn_failure_time is not None:

        failure_time_error = (

            pinn_failure_time
            - reference_failure_time

        )

        absolute_failure_error = abs(
            failure_time_error
        )


        print(
            f"PINN failure-time error: "
            f"{failure_time_error:+.3f} days"
        )

        print(
            f"Absolute failure-time error: "
            f"{absolute_failure_error:.3f} days"
        )




plt.figure(
    figsize=(9, 5)
)




plt.plot(

    t,

    pinn_damage,

    linewidth=2,

    label="PINN Predicted Degradation",
)




plt.plot(

    t,

    true_damage,

    "--",

    linewidth=1.5,

    label="Reference Degradation",
)




plt.axhline(

    FAILURE_THRESHOLD,

    linestyle="--",

    label=(
        f"Assumed Failure Threshold "
        f"= {FAILURE_THRESHOLD:.2f}"
    ),
)




if pinn_failure_time is not None:

    plt.axvline(

        pinn_failure_time,

        linestyle=":",

        label=(
            f"PINN Failure Time "
            f"= {pinn_failure_time:.2f} d"
        ),
    )


    plt.scatter(

        [pinn_failure_time],

        [FAILURE_THRESHOLD],

        s=80,

        zorder=5,
    )




if CURRENT_TIME_DAYS > 0:

    plt.axvline(

        CURRENT_TIME_DAYS,

        linestyle="-.",

        label=(
            f"Current Time "
            f"= {CURRENT_TIME_DAYS:.2f} d"
        ),
    )


plt.xlabel(
    "Time (days)"
)

plt.ylabel(
    "Normalized Degradation Index"
)

plt.title(
    "Remaining Useful Life Estimation "
    "Using Physics-Informed Neural Network"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()