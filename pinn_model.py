

import torch
import torch.nn as nn



class PINN(nn.Module):

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
            torch.tensor(float(t_min))
        )

        self.register_buffer(
            "t_max",
            torch.tensor(float(t_max))
        )

        self.register_buffer(
            "T_min",
            torch.tensor(float(T_min))
        )

        self.register_buffer(
            "T_max",
            torch.tensor(float(T_max))
        )

        self.register_buffer(
            "D_min",
            torch.tensor(float(D_min))
        )

        self.register_buffer(
            "D_max",
            torch.tensor(float(D_max))
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


    def forward(self, t, T, D):



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




def degradation_physics_loss(
    model,
    t,
    T,
    D,
    dose_rate,
    thermal_acceleration,
    total_thermal_exposure,
    radiation_reference_dose,
    w_thermal,
    w_radiation,
):




    damage_pred = model(
        t,
        T,
        D,
    )




    dt = (
        t[1:]
        - t[:-1]
    )

    predicted_rate = (
        damage_pred[1:]
        - damage_pred[:-1]
    ) / dt




    thermal_rate = (
        thermal_acceleration
        /
        total_thermal_exposure
    )

    radiation_rate = (
        dose_rate
        /
        radiation_reference_dose
    )

    physical_rate = (
        w_thermal * thermal_rate
        +
        w_radiation * radiation_rate
    )



    physical_rate_mid = (
        0.5
        * (
            physical_rate[1:]
            + physical_rate[:-1]
        )
    )




    residual = (
        predicted_rate
        - physical_rate_mid
    )

    physics_loss = torch.mean(
        residual**2
    )

    return physics_loss




