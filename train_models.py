

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn



from degradation_model import (
    t_deg,
    T_deg,
    damage_deg,
    thermal_acceleration,
    thermal_exposure_days,
    radiation_reference_dose,
    W_THERMAL,
    W_RADIATION,
)

from radiation_sim import (
    D_rad,
    dose_rate,
)

from pinn_model import (
    PINN,
    degradation_physics_loss,
)




SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)




N_TRAIN_POINTS = 180

N_EPOCHS = 4000

LEARNING_RATE = 1.0e-3


# hi nice to meet u

LAMBDA_PHYSICS = 0.25

LAMBDA_INITIAL = 5.0




t_np = np.asarray(
    t_deg,
    dtype=np.float32,
)

T_np = np.asarray(
    T_deg,
    dtype=np.float32,
)

D_np = np.asarray(
    D_rad,
    dtype=np.float32,
)

damage_np = np.asarray(
    damage_deg,
    dtype=np.float32,
)

dose_rate_np = np.asarray(
    dose_rate,
    dtype=np.float32,
)

thermal_acceleration_np = np.asarray(
    thermal_acceleration,
    dtype=np.float32,
)




train_indices = np.unique(
    np.linspace(
        0,
        len(t_np) - 1,
        N_TRAIN_POINTS,
        dtype=int,
    )
)




test_mask = np.ones(
    len(t_np),
    dtype=bool,
)

test_mask[train_indices] = False

test_indices = np.where(
    test_mask
)[0]


print(
    f"\nTotal trajectory points: {len(t_np)}"
)

print(
    f"Training degradation observations: "
    f"{len(train_indices)}"
)

print(
    f"Held-out test points: "
    f"{len(test_indices)}"
)



def column_tensor(array):


    return torch.tensor(
        array,
        dtype=torch.float32,
    ).reshape(-1, 1)




t_train = column_tensor(
    t_np[train_indices]
)

T_train = column_tensor(
    T_np[train_indices]
)

D_train = column_tensor(
    D_np[train_indices]
)

damage_train = column_tensor(
    damage_np[train_indices]
)




t_phys = column_tensor(
    t_np
)

T_phys = column_tensor(
    T_np
)

D_phys = column_tensor(
    D_np
)

dose_rate_phys = column_tensor(
    dose_rate_np
)

thermal_acceleration_phys = column_tensor(
    thermal_acceleration_np
)




t0 = column_tensor(
    [t_np[0]]
)

T0 = column_tensor(
    [T_np[0]]
)

D0 = column_tensor(
    [D_np[0]]
)

damage0 = column_tensor(
    [damage_np[0]]
)


# whats a good name that i can call a golden fish ?

class TimeOnlyNN(nn.Module):

    def __init__(
        self,
        t_min,
        t_max,
    ):

        super().__init__()

        self.register_buffer(
            "t_min",
            torch.tensor(
                float(t_min)
            ),
        )

        self.register_buffer(
            "t_max",
            torch.tensor(
                float(t_max)
            ),
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
            (
                self.t_max
                - self.t_min
            )
        )

        return self.net(
            tn
        )


# ich kann nicht

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
            torch.tensor(
                float(t_min)
            ),
        )

        self.register_buffer(
            "t_max",
            torch.tensor(
                float(t_max)
            ),
        )

        self.register_buffer(
            "T_min",
            torch.tensor(
                float(T_min)
            ),
        )

        self.register_buffer(
            "T_max",
            torch.tensor(
                float(T_max)
            ),
        )

        self.register_buffer(
            "D_min",
            torch.tensor(
                float(D_min)
            ),
        )

        self.register_buffer(
            "D_max",
            torch.tensor(
                float(D_max)
            ),
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
            (
                self.t_max
                - self.t_min
            )
        )

        Tn = (
            (T - self.T_min)
            /
            (
                self.T_max
                - self.T_min
            )
        )

        Dn = (
            (D - self.D_min)
            /
            (
                self.D_max
                - self.D_min
            )
        )


        x = torch.cat(
            [
                tn,
                Tn,
                Dn,
            ],
            dim=1,
        )


        return self.net(
            x
        )


# BOOOOOOOOOOOO

time_model = TimeOnlyNN(

    t_min=t_np.min(),

    t_max=t_np.max(),
)


environment_model = EnvironmentalNN(

    t_min=t_np.min(),
    t_max=t_np.max(),

    T_min=T_np.min(),
    T_max=T_np.max(),

    D_min=D_np.min(),
    D_max=D_np.max(),
)


pinn_model = PINN(

    t_min=t_np.min(),
    t_max=t_np.max(),

    T_min=T_np.min(),
    T_max=T_np.max(),

    D_min=D_np.min(),
    D_max=D_np.max(),
)




optimizer_time = torch.optim.Adam(

    time_model.parameters(),

    lr=LEARNING_RATE,
)


optimizer_environment = torch.optim.Adam(

    environment_model.parameters(),

    lr=LEARNING_RATE,
)


optimizer_pinn = torch.optim.Adam(

    pinn_model.parameters(),

    lr=LEARNING_RATE,
)


mse_loss = nn.MSELoss()


# what are u doing here amigo ?

time_loss_history = []

environment_loss_history = []

pinn_total_history = []

pinn_data_history = []

pinn_physics_history = []




print(
    "\nTraining Time-Only NN..."
)


for epoch in range(
    N_EPOCHS
):

    optimizer_time.zero_grad()


    prediction = time_model(
        t_train
    )


    loss = mse_loss(
        prediction,
        damage_train,
    )


    loss.backward()

    optimizer_time.step()


    time_loss_history.append(
        loss.item()
    )


    if epoch % 500 == 0:

        print(
            f"[Time NN] "
            f"Epoch {epoch:4d} | "
            f"Loss = "
            f"{loss.item():.6e}"
        )




print(
    "\nTraining Environmental NN..."
)


for epoch in range(
    N_EPOCHS
):

    optimizer_environment.zero_grad()


    prediction = environment_model(

        t_train,

        T_train,

        D_train,
    )


    loss = mse_loss(

        prediction,

        damage_train,
    )


    loss.backward()

    optimizer_environment.step()


    environment_loss_history.append(
        loss.item()
    )


    if epoch % 500 == 0:

        print(
            f"[Environment NN] "
            f"Epoch {epoch:4d} | "
            f"Loss = "
            f"{loss.item():.6e}"
        )



print(
    "\nTraining Physics-Informed NN..."
)


for epoch in range(
    N_EPOCHS
):

    optimizer_pinn.zero_grad()



    prediction_train = pinn_model(

        t_train,

        T_train,

        D_train,
    )


    data_loss = mse_loss(

        prediction_train,

        damage_train,
    )




    physics_loss = degradation_physics_loss(

        model=pinn_model,

        t=t_phys,

        T=T_phys,

        D=D_phys,

        dose_rate=dose_rate_phys,

        thermal_acceleration=(
            thermal_acceleration_phys
        ),

        total_thermal_exposure=float(
            thermal_exposure_days[-1]
        ),

        radiation_reference_dose=float(
            radiation_reference_dose
        ),

        w_thermal=W_THERMAL,

        w_radiation=W_RADIATION,
    )




    initial_prediction = pinn_model(

        t0,

        T0,

        D0,
    )


    initial_loss = mse_loss(

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


    pinn_total_history.append(
        total_loss.item()
    )

    pinn_data_history.append(
        data_loss.item()
    )

    pinn_physics_history.append(
        physics_loss.item()
    )


    if epoch % 500 == 0:

        print(
            f"[PINN] "
            f"Epoch {epoch:4d} | "
            f"Total = "
            f"{total_loss.item():.6e} | "
            f"Data = "
            f"{data_loss.item():.6e} | "
            f"Physics = "
            f"{physics_loss.item():.6e}"
        )



t_all = column_tensor(
    t_np
)

T_all = column_tensor(
    T_np
)

D_all = column_tensor(
    D_np
)




time_model.eval()

environment_model.eval()

pinn_model.eval()


with torch.no_grad():

    damage_pred_time = (

        time_model(
            t_all
        )

        .cpu()

        .numpy()

        .flatten()
    )


    damage_pred_environment = (

        environment_model(

            t_all,

            T_all,

            D_all,
        )

        .cpu()

        .numpy()

        .flatten()
    )


    damage_pred_pinn = (

        pinn_model(

            t_all,

            T_all,

            D_all,
        )

        .cpu()

        .numpy()

        .flatten()
    )



def rmse(
    true,
    predicted,
):

    return np.sqrt(

        np.mean(

            (
                true
                - predicted
            ) ** 2

        )
    )


def mae(
    true,
    predicted,
):

    return np.mean(

        np.abs(

            true
            - predicted

        )
    )


true_test = damage_np[
    test_indices
]



time_rmse = rmse(

    true_test,

    damage_pred_time[
        test_indices
    ],
)


time_mae = mae(

    true_test,

    damage_pred_time[
        test_indices
    ],
)




environment_rmse = rmse(

    true_test,

    damage_pred_environment[
        test_indices
    ],
)


environment_mae = mae(

    true_test,

    damage_pred_environment[
        test_indices
    ],
)




pinn_rmse = rmse(

    true_test,

    damage_pred_pinn[
        test_indices
    ],
)


pinn_mae = mae(

    true_test,

    damage_pred_pinn[
        test_indices
    ],
)




print(
    "\n======================================"
)

print(
    "HELD-OUT TEST RESULTS"
)

print(
    "======================================"
)


print(

    f"Time-only NN      "

    f"RMSE: {time_rmse:.6f} "

    f"| MAE: {time_mae:.6f}"
)


print(

    f"Environmental NN  "

    f"RMSE: {environment_rmse:.6f} "

    f"| MAE: {environment_mae:.6f}"
)


print(

    f"Physics PINN      "

    f"RMSE: {pinn_rmse:.6f} "

    f"| MAE: {pinn_mae:.6f}"
)



OUTPUT_DIR = Path(
    "outputs"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


np.savez(

    OUTPUT_DIR
    / "model_predictions.npz",

    time=t_np,

    temperature=T_np,

    cumulative_dose=D_np,

    dose_rate=dose_rate_np,

    true_damage=damage_np,

    time_nn=damage_pred_time,

    environment_nn=damage_pred_environment,

    pinn=damage_pred_pinn,

    train_indices=train_indices,

    test_indices=test_indices,

    time_rmse=time_rmse,

    environment_rmse=(
        environment_rmse
    ),

    pinn_rmse=pinn_rmse,

    time_mae=time_mae,

    environment_mae=(
        environment_mae
    ),

    pinn_mae=pinn_mae,
)


torch.save(

    time_model.state_dict(),

    OUTPUT_DIR
    / "time_nn.pt",
)


torch.save(

    environment_model.state_dict(),

    OUTPUT_DIR
    / "environment_nn.pt",
)


torch.save(

    pinn_model.state_dict(),

    OUTPUT_DIR
    / "physics_pinn.pt",
)




plt.figure(
    figsize=(10, 6)
)


plt.plot(

    t_np,

    damage_np,

    linewidth=2.5,

    label="Reference Degradation",
)


plt.scatter(

    t_np[
        train_indices
    ],

    damage_np[
        train_indices
    ],

    s=18,

    label=(
        "Sparse Training Observations"
    ),
)


plt.plot(

    t_np,

    damage_pred_time,

    "--",

    label="Time-Only NN",
)


plt.plot(

    t_np,

    damage_pred_environment,

    "--",

    label="Environmental NN",
)


plt.plot(

    t_np,

    damage_pred_pinn,

    linewidth=2,

    label="Physics-Informed NN",
)


plt.xlabel(
    "Time (days)"
)

plt.ylabel(
    "Normalized Degradation Index"
)

plt.title(
    "Degradation Prediction Model Comparison"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()




plt.figure(
    figsize=(9, 5)
)


plt.plot(

    t_np,

    damage_np,

    linewidth=2,

    label="Reference Degradation",
)


plt.plot(

    t_np,

    damage_pred_pinn,

    linewidth=2,

    label="PINN Prediction",
)


plt.scatter(

    t_np[
        train_indices
    ],

    damage_np[
        train_indices
    ],

    s=18,

    label="Training Observations",
)


plt.xlabel(
    "Time (days)"
)

plt.ylabel(
    "Normalized Degradation Index"
)

plt.title(
    "Physics-Informed Neural Network "
    "Degradation Prediction"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()




plt.figure(
    figsize=(9, 5)
)


plt.semilogy(

    pinn_data_history,

    label="PINN Data Loss",
)


plt.semilogy(

    pinn_physics_history,

    label="PINN Physics Loss",
)


plt.semilogy(

    pinn_total_history,

    label="PINN Total Loss",
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "PINN Training Loss History"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()