

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from environment_data import radiation

from degradation_model import (
    t_deg,
    T_deg,
    thermal_damage,
    thermal_acceleration,
    thermal_exposure_days,
    W_THERMAL,
    W_RADIATION,
)

from pinn_model import (
    PINN,
    degradation_physics_loss,
)




SEED = 42

rng = np.random.default_rng(SEED)



N_MONTE_CARLO = 50


# bye bye bye



CURRENT_TIME_DAYS = 8.0


# hi

N_OBSERVATIONS = 25




MEASUREMENT_NOISE_STD = 0.01




N_EPOCHS = 1500

LEARNING_RATE = 1.0e-3

LAMBDA_PHYSICS = 0.25

LAMBDA_INITIAL = 5.0





FAILURE_THRESHOLD = 0.85




t_np = np.asarray(
    t_deg,
    dtype=np.float32,
)

T_np = np.asarray(
    T_deg,
    dtype=np.float32,
)

thermal_damage_np = np.asarray(
    thermal_damage,
    dtype=np.float32,
)

thermal_acceleration_np = np.asarray(
    thermal_acceleration,
    dtype=np.float32,
)


DAYS_TOTAL = float(
    radiation["days_in_model_month"]
)

MONTHLY_DOSE_RAD = float(
    radiation["monthly_dose_rad"]
)

STORM_FACTOR_MIN = float(
    radiation["storm_factor_min"]
)

STORM_FACTOR_MAX = float(
    radiation["storm_factor_max"]
)



STORM_DURATION_DAYS = 1.0

STORM_START = (
    radiation["storm_day"]
    - STORM_DURATION_DAYS / 2.0
)

STORM_END = (
    radiation["storm_day"]
    + STORM_DURATION_DAYS / 2.0
)



def column_tensor(array):

    return torch.tensor(
        array,
        dtype=torch.float32,
    ).reshape(-1, 1)




class TimeOnlyNN(nn.Module):

    def __init__(
        self,
        t_min,
        t_max,
    ):

        super().__init__()

        self.register_buffer(
            "t_min",
            torch.tensor(float(t_min)),
        )

        self.register_buffer(
            "t_max",
            torch.tensor(float(t_max)),
        )

        self.net = nn.Sequential(

            nn.Linear(1, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 1),
        )


    def forward(self, t):

        tn = (
            (t - self.t_min)
            /
            (self.t_max - self.t_min)
        )

        return self.net(tn)




class EnvironmentalNN(nn.Module):

    def __init__(
        self,
        t_min,
        t_max,
        T_min,
        T_max,
        D_min,
        D_max,
    ):

        super().__init__()

        self.register_buffer(
            "t_min",
            torch.tensor(float(t_min)),
        )

        self.register_buffer(
            "t_max",
            torch.tensor(float(t_max)),
        )

        self.register_buffer(
            "T_min",
            torch.tensor(float(T_min)),
        )

        self.register_buffer(
            "T_max",
            torch.tensor(float(T_max)),
        )

        self.register_buffer(
            "D_min",
            torch.tensor(float(D_min)),
        )

        self.register_buffer(
            "D_max",
            torch.tensor(float(D_max)),
        )


        self.net = nn.Sequential(

            nn.Linear(3, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 1),
        )


    def forward(
        self,
        t,
        T,
        D,
    ):

        tn = (
            (t - self.t_min)
            /
            (self.t_max - self.t_min)
        )

        Tn = (
            (T - self.T_min)
            /
            (self.T_max - self.T_min)
        )

        Dn = (
            (D - self.D_min)
            /
            (self.D_max - self.D_min)
        )


        x = torch.cat(
            [tn, Tn, Dn],
            dim=1,
        )

        return self.net(x)




def build_radiation_profile(
    storm_factor,
):



    quiet_rate = (

        MONTHLY_DOSE_RAD

        /

        (
            DAYS_TOTAL

            +

            (storm_factor - 1.0)
            * STORM_DURATION_DAYS
        )
    )


    storm_rate = (
        storm_factor
        * quiet_rate
    )


    dose_rate = np.full_like(
        t_np,
        quiet_rate,
        dtype=np.float64,
    )


    storm_mask = (

        (t_np >= STORM_START)

        &

        (t_np < STORM_END)
    )


    dose_rate[
        storm_mask
    ] = storm_rate




    increments = (

        0.5

        *

        (
            dose_rate[1:]
            +
            dose_rate[:-1]
        )

        *

        np.diff(t_np)
    )


    cumulative_dose = np.concatenate(

        (
            [0.0],

            np.cumsum(increments),
        )
    )


    scale = (
        MONTHLY_DOSE_RAD
        /
        cumulative_dose[-1]
    )


    cumulative_dose *= scale

    dose_rate *= scale


    return (

        cumulative_dose.astype(
            np.float32
        ),

        dose_rate.astype(
            np.float32
        ),
    )




def build_degradation(
    cumulative_dose,
):
    


    radiation_damage = (

        cumulative_dose

        /

        MONTHLY_DOSE_RAD
    )


    damage = (

        W_THERMAL
        * thermal_damage_np

        +

        W_RADIATION
        * radiation_damage
    )


    return np.asarray(
        damage,
        dtype=np.float32,
    )




def find_threshold_crossing(
    time,
    damage,
    threshold,
):

    time = np.asarray(
        time
    ).flatten()

    damage = np.asarray(
        damage
    ).flatten()


    indices = np.where(
        damage >= threshold
    )[0]


    if len(indices) == 0:

        return np.nan


    upper = indices[0]


    if upper == 0:

        return float(
            time[0]
        )


    lower = upper - 1


    t1 = time[lower]
    t2 = time[upper]

    d1 = damage[lower]
    d2 = damage[upper]


    if np.isclose(
        d1,
        d2,
    ):

        return float(t2)


    crossing = (

        t1

        +

        (
            threshold - d1
        )

        /

        (
            d2 - d1
        )

        *

        (
            t2 - t1
        )
    )


    return float(
        crossing
    )




def rmse(
    true,
    predicted,
):

    return float(

        np.sqrt(

            np.mean(

                (
                    true
                    - predicted
                ) ** 2
            )
        )
    )




def choose_training_indices(
    local_rng,
):



    available = np.where(

        t_np
        <= CURRENT_TIME_DAYS

    )[0]


    first_index = available[0]

    last_index = available[-1]


    interior = available[
        1:-1
    ]


    n_interior = (
        N_OBSERVATIONS
        - 2
    )


    selected = local_rng.choice(

        interior,

        size=n_interior,

        replace=False,
    )


    indices = np.concatenate(

        (
            [first_index],

            selected,

            [last_index],
        )
    )


    return np.sort(
        indices
    )




def run_one_realization(
    run_number,
    storm_factor,
    local_rng,
):




    D_sim, dose_rate_sim = (
        build_radiation_profile(
            storm_factor
        )
    )


    true_damage = (
        build_degradation(
            D_sim
        )
    )




    train_indices = (
        choose_training_indices(
            local_rng
        )
    )


    # HELLO THERE! ( Obi one Kneboi)

    measurement_noise = (
        local_rng.normal(

            loc=0.0,

            scale=MEASUREMENT_NOISE_STD,

            size=len(train_indices),
        )
    )


    observed_damage = (

        true_damage[
            train_indices
        ]

        +

        measurement_noise
    )


    observed_damage = np.clip(

        observed_damage,

        0.0,

        1.0,
    ).astype(np.float32)



    observed_damage[0] = (
        true_damage[
            train_indices[0]
        ]
    )




    future_mask = (
        t_np
        > CURRENT_TIME_DAYS
    )




    t_train = column_tensor(
        t_np[
            train_indices
        ]
    )

    T_train = column_tensor(
        T_np[
            train_indices
        ]
    )

    D_train = column_tensor(
        D_sim[
            train_indices
        ]
    )

    damage_train = column_tensor(
        observed_damage
    )






    N_COLLOCATION = 601


    collocation_indices = np.unique(

        np.linspace(

            0,

            len(t_np) - 1,

            N_COLLOCATION,

            dtype=int,
        )
    )


    t_phys = column_tensor(
        t_np[
            collocation_indices
        ]
    )

    T_phys = column_tensor(
        T_np[
            collocation_indices
        ]
    )

    D_phys = column_tensor(
        D_sim[
            collocation_indices
        ]
    )

    dose_rate_phys = column_tensor(
        dose_rate_sim[
            collocation_indices
        ]
    )

    thermal_acceleration_phys = column_tensor(

        thermal_acceleration_np[
            collocation_indices
        ]
    )




    t_all = column_tensor(
        t_np
    )

    T_all = column_tensor(
        T_np
    )

    D_all = column_tensor(
        D_sim
    )




    t0 = column_tensor(
        [t_np[0]]
    )

    T0 = column_tensor(
        [T_np[0]]
    )

    D0 = column_tensor(
        [D_sim[0]]
    )

    damage0 = column_tensor(
        [true_damage[0]]
    )




    torch.manual_seed(
        SEED
        + run_number
    )


    time_model = TimeOnlyNN(

        t_min=t_np.min(),

        t_max=t_np.max(),
    )


    environment_model = EnvironmentalNN(

        t_min=t_np.min(),
        t_max=t_np.max(),

        T_min=T_np.min(),
        T_max=T_np.max(),

        D_min=0.0,
        D_max=MONTHLY_DOSE_RAD,
    )


    pinn_model = PINN(

        t_min=t_np.min(),
        t_max=t_np.max(),

        T_min=T_np.min(),
        T_max=T_np.max(),

        D_min=0.0,
        D_max=MONTHLY_DOSE_RAD,
    )


    loss_fn = nn.MSELoss()




    optimizer_time = torch.optim.Adam(

        time_model.parameters(),

        lr=LEARNING_RATE,
    )


    for _ in range(
        N_EPOCHS
    ):

        optimizer_time.zero_grad()


        prediction = time_model(
            t_train
        )


        loss = loss_fn(

            prediction,

            damage_train,
        )


        loss.backward()

        optimizer_time.step()




    optimizer_environment = torch.optim.Adam(

        environment_model.parameters(),

        lr=LEARNING_RATE,
    )


    for _ in range(
        N_EPOCHS
    ):

        optimizer_environment.zero_grad()


        prediction = environment_model(

            t_train,

            T_train,

            D_train,
        )


        loss = loss_fn(

            prediction,

            damage_train,
        )


        loss.backward()

        optimizer_environment.step()


    # do u love sweetes?

    optimizer_pinn = torch.optim.Adam(

        pinn_model.parameters(),

        lr=LEARNING_RATE,
    )


    for _ in range(
        N_EPOCHS
    ):

        optimizer_pinn.zero_grad()




        prediction_train = pinn_model(

            t_train,

            T_train,

            D_train,
        )


        data_loss = loss_fn(

            prediction_train,

            damage_train,
        )




        physics_loss = (
            degradation_physics_loss(

                model=pinn_model,

                t=t_phys,

                T=T_phys,

                D=D_phys,

                dose_rate=(
                    dose_rate_phys
                ),

                thermal_acceleration=(
                    thermal_acceleration_phys
                ),

                total_thermal_exposure=float(
                    thermal_exposure_days[-1]
                ),

                radiation_reference_dose=(
                    MONTHLY_DOSE_RAD
                ),

                w_thermal=(
                    W_THERMAL
                ),

                w_radiation=(
                    W_RADIATION
                ),
            )
        )




        initial_prediction = pinn_model(

            t0,

            T0,

            D0,
        )


        initial_loss = loss_fn(

            initial_prediction,

            damage0,
        )




        total_loss = (

            data_loss

            +

            LAMBDA_PHYSICS
            * physics_loss

            +

            LAMBDA_INITIAL
            * initial_loss
        )


        total_loss.backward()

        optimizer_pinn.step()


    #  drink some tea while readin this

    time_model.eval()

    environment_model.eval()

    pinn_model.eval()


    with torch.no_grad():


        pred_time = (

            time_model(
                t_all
            )

            .cpu()

            .numpy()

            .flatten()
        )


        pred_environment = (

            environment_model(

                t_all,

                T_all,

                D_all,
            )

            .cpu()

            .numpy()

            .flatten()
        )


        pred_pinn = (

            pinn_model(

                t_all,

                T_all,

                D_all,
            )

            .cpu()

            .numpy()

            .flatten()
        )




    future_true = true_damage[
        future_mask
    ]


    rmse_time = rmse(

        future_true,

        pred_time[
            future_mask
        ],
    )


    rmse_environment = rmse(

        future_true,

        pred_environment[
            future_mask
        ],
    )


    rmse_pinn = rmse(

        future_true,

        pred_pinn[
            future_mask
        ],
    )




    failure_true = (
        find_threshold_crossing(

            t_np,

            true_damage,

            FAILURE_THRESHOLD,
        )
    )


    failure_time = (
        find_threshold_crossing(

            t_np,

            pred_time,

            FAILURE_THRESHOLD,
        )
    )


    failure_environment = (
        find_threshold_crossing(

            t_np,

            pred_environment,

            FAILURE_THRESHOLD,
        )
    )


    failure_pinn = (
        find_threshold_crossing(

            t_np,

            pred_pinn,

            FAILURE_THRESHOLD,
        )
    )




    if (
        np.isfinite(
            failure_true
        )

        and

        failure_true
        > CURRENT_TIME_DAYS
    ):

        true_rul = (

            failure_true
            - CURRENT_TIME_DAYS
        )

    else:

        true_rul = np.nan




    def failure_to_rul(
        failure_time,
    ):

        if not np.isfinite(
            failure_time
        ):

            return np.nan


        if failure_time <= CURRENT_TIME_DAYS:

            return np.nan


        return (
            failure_time
            - CURRENT_TIME_DAYS
        )


    time_rul = failure_to_rul(
        failure_time
    )

    environment_rul = failure_to_rul(
        failure_environment
    )

    pinn_rul = failure_to_rul(
        failure_pinn
    )




    def absolute_rul_error(
        predicted,
        true,
    ):

        if (
            np.isfinite(predicted)

            and

            np.isfinite(true)
        ):

            return abs(
                predicted
                - true
            )


        return np.nan


    time_rul_error = (
        absolute_rul_error(

            time_rul,

            true_rul,
        )
    )


    environment_rul_error = (
        absolute_rul_error(

            environment_rul,

            true_rul,
        )
    )


    pinn_rul_error = (
        absolute_rul_error(

            pinn_rul,

            true_rul,
        )
    )


    return {

        "storm_factor":
            storm_factor,

        "true_rul":
            true_rul,

        "time_rul":
            time_rul,

        "environment_rul":
            environment_rul,

        "pinn_rul":
            pinn_rul,

        "rmse_time":
            rmse_time,

        "rmse_environment":
            rmse_environment,

        "rmse_pinn":
            rmse_pinn,

        "rul_error_time":
            time_rul_error,

        "rul_error_environment":
            environment_rul_error,

        "rul_error_pinn":
            pinn_rul_error,
    }




results = []


print(
    "\n======================================"
)

print(
    "MONTE CARLO PRE-STORM RUL VALIDATION"
)

print(
    "======================================"
)

print(
    f"Runs: {N_MONTE_CARLO}"
)

print(
    f"Current operating time: "
    f"{CURRENT_TIME_DAYS:.1f} days"
)

print(
    f"Available degradation measurements: "
    f"{N_OBSERVATIONS}"
)

print(
    f"Measurement-noise standard deviation: "
    f"{MEASUREMENT_NOISE_STD:.3f}"
)

print(
    f"Storm-factor range: "
    f"{STORM_FACTOR_MIN:.0f}-"
    f"{STORM_FACTOR_MAX:.0f}x"
)

print()


for run in range(
    N_MONTE_CARLO
):




    storm_factor = rng.uniform(

        STORM_FACTOR_MIN,

        STORM_FACTOR_MAX,
    )


    run_rng = np.random.default_rng(

        SEED
        + 10_000
        + run
    )


    result = (
        run_one_realization(

            run_number=run,

            storm_factor=storm_factor,

            local_rng=run_rng,
        )
    )


    results.append(
        result
    )


    if (
        (run + 1) % 5 == 0

        or

        run == 0
    ):

        print(

            f"Run "
            f"{run + 1:3d}/"
            f"{N_MONTE_CARLO}"

            f" | Storm = "
            f"{storm_factor:6.2f}x"

            f" | PINN RMSE = "
            f"{result['rmse_pinn']:.5f}"
        )


print(
    "\nMonte Carlo finished."
)




def result_array(
    key,
):

    return np.asarray(
        [
            item[key]
            for item in results
        ],
        dtype=float,
    )


storm_factors = result_array(
    "storm_factor"
)

true_rul = result_array(
    "true_rul"
)


time_rul = result_array(
    "time_rul"
)

environment_rul = result_array(
    "environment_rul"
)

pinn_rul = result_array(
    "pinn_rul"
)


rmse_time = result_array(
    "rmse_time"
)

rmse_environment = result_array(
    "rmse_environment"
)

rmse_pinn = result_array(
    "rmse_pinn"
)


rul_error_time = result_array(
    "rul_error_time"
)

rul_error_environment = result_array(
    "rul_error_environment"
)

rul_error_pinn = result_array(
    "rul_error_pinn"
)




def summarize(
    name,
    values,
):

    finite = values[
        np.isfinite(
            values
        )
    ]


    if len(finite) == 0:

        print(
            f"{name:<24} "
            f"No valid predictions"
        )

        return


    print(

        f"{name:<24}"

        f"Mean = "
        f"{np.mean(finite):.6f}"

        f" | Std = "
        f"{np.std(finite):.6f}"

        f" | Median = "
        f"{np.median(finite):.6f}"
    )




print(
    "\n======================================"
)

print(
    "FUTURE DEGRADATION RMSE"
)

print(
    "======================================"
)


summarize(
    "Time-only NN",
    rmse_time,
)

summarize(
    "Environmental NN",
    rmse_environment,
)

summarize(
    "Physics PINN",
    rmse_pinn,
)




print(
    "\n======================================"
)

print(
    "ABSOLUTE RUL ERROR [days]"
)

print(
    "======================================"
)


summarize(
    "Time-only NN",
    rul_error_time,
)

summarize(
    "Environmental NN",
    rul_error_environment,
)

summarize(
    "Physics PINN",
    rul_error_pinn,
)




print(
    "\n======================================"
)

print(
    "VALID FUTURE-FAILURE PREDICTIONS"
)

print(
    "======================================"
)


print(

    f"Time-only NN: "
    f"{np.sum(np.isfinite(time_rul))}/"
    f"{N_MONTE_CARLO}"
)


print(

    f"Environmental NN: "
    f"{np.sum(np.isfinite(environment_rul))}/"
    f"{N_MONTE_CARLO}"
)


print(

    f"Physics PINN: "
    f"{np.sum(np.isfinite(pinn_rul))}/"
    f"{N_MONTE_CARLO}"
)




OUTPUT_DIR = Path(
    "outputs"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


np.savez(

    OUTPUT_DIR
    / "monte_carlo_results.npz",

    storm_factor=storm_factors,

    true_rul=true_rul,

    time_rul=time_rul,

    environment_rul=environment_rul,

    pinn_rul=pinn_rul,

    rmse_time=rmse_time,

    rmse_environment=rmse_environment,

    rmse_pinn=rmse_pinn,

    rul_error_time=rul_error_time,

    rul_error_environment=(
        rul_error_environment
    ),

    rul_error_pinn=rul_error_pinn,

    current_time_days=(
        CURRENT_TIME_DAYS
    ),

    failure_threshold=(
        FAILURE_THRESHOLD
    ),

    measurement_noise_std=(
        MEASUREMENT_NOISE_STD
    ),
)




plt.figure(
    figsize=(9, 5)
)


plt.boxplot(

    [
        rmse_time,

        rmse_environment,

        rmse_pinn,
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

plt.tight_layout()

plt.show()


# what are u doing ?

valid_time_error = rul_error_time[
    np.isfinite(
        rul_error_time
    )
]

valid_environment_error = (
    rul_error_environment[
        np.isfinite(
            rul_error_environment
        )
    ]
)

valid_pinn_error = rul_error_pinn[
    np.isfinite(
        rul_error_pinn
    )
]


plt.figure(
    figsize=(9, 5)
)


plt.boxplot(

    [
        valid_time_error,

        valid_environment_error,

        valid_pinn_error,
    ],

    tick_labels=[

        "Time-Only NN",

        "Environmental NN",

        "Physics PINN",
    ],
)


plt.ylabel(
    "Absolute RUL Error (days)"
)

plt.title(
    "Monte Carlo RUL Prediction Error "
    "(Pre-Storm Observations)"
)

plt.grid(
    True,
    axis="y",
)

plt.tight_layout()

plt.show()




plt.figure(
    figsize=(9, 5)
)


plt.scatter(

    storm_factors,

    rmse_environment,

    label="Environmental NN",
)


plt.scatter(

    storm_factors,

    rmse_pinn,

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

plt.tight_layout()

plt.show()