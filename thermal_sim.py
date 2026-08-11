

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legval

from environment_data import (
    orbit,
    thermal,
    CERES_OLR_COEFF,
    CERES_ALBEDO_COEFF,
)





EARTH_RADIUS_KM = 6378.137


SIDEREAL_DAY_S = 86164.0905


DT_TH = 60.0
GRID_STEP_DEG = 2.0
N_ORBITS = 3


INITIAL_TEMPERATURE_K = 280.0




def extraterrestrial_solar_flux(day_of_year):


    G_sc = thermal["solar_constant_W_m2"]

    return G_sc * (
        1.0
        + 0.033
        * np.cos(np.deg2rad(360.0 * day_of_year / 365.0))
    )


G_ON = extraterrestrial_solar_flux(
    thermal["day_of_year"]
)




def seasonal_coefficients(table, day_of_year):


    omega = 2.0 * np.pi / 365.0
    tau = day_of_year - 1.0

    return (
        table[:, 0]
        + table[:, 1] * np.cos(omega * tau)
        + table[:, 2] * np.sin(omega * tau)
    )


OLR_COEFF = seasonal_coefficients(
    CERES_OLR_COEFF,
    thermal["day_of_year"],
)

ALBEDO_COEFF = seasonal_coefficients(
    CERES_ALBEDO_COEFF,
    thermal["day_of_year"],
)




def zonal_field(latitude_deg, coefficients):


    latitude_deg = np.asarray(latitude_deg, dtype=float)

    x = np.sin(np.deg2rad(latitude_deg))

    result = np.zeros_like(x, dtype=float)

    for degree, coefficient in enumerate(coefficients):

        # Legendre polynomial P_degree(x)
        polynomial = [0.0] * degree + [1.0]

        P_l = legval(x, polynomial)

        Y_l0 = np.sqrt(
            (2.0 * degree + 1.0) / (4.0 * np.pi)
        ) * P_l

        result += coefficient * Y_l0

    return result



latitudes = np.arange(
    -90.0 + GRID_STEP_DEG / 2.0,
    90.0,
    GRID_STEP_DEG,
)

longitudes = np.arange(
    GRID_STEP_DEG / 2.0,
    360.0,
    GRID_STEP_DEG,
)

LAT, LON = np.meshgrid(
    latitudes,
    longitudes,
    indexing="ij",
)

lat_flat = LAT.ravel()
lon_flat = LON.ravel()

lat_rad = np.deg2rad(lat_flat)
lon_rad = np.deg2rad(lon_flat)

dlat = np.deg2rad(GRID_STEP_DEG)
dlon = np.deg2rad(GRID_STEP_DEG)


earth_points = EARTH_RADIUS_KM * np.column_stack(
    (
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad),
    )
)


earth_normals = earth_points / EARTH_RADIUS_KM



surface_area = (
    EARTH_RADIUS_KM**2
    * np.cos(lat_rad)
    * dlat
    * dlon
)


OLR_GRID = zonal_field(
    lat_flat,
    OLR_COEFF,
)

ALBEDO_GRID = zonal_field(
    lat_flat,
    ALBEDO_COEFF,
)

ALBEDO_GRID = np.clip(
    ALBEDO_GRID,
    0.0,
    1.0,
)




def rotation_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def rotation_x(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s,  c],
        ]
    )




def solve_kepler(mean_anomaly, eccentricity):
    """
    Solve:

        M = E - e*sin(E)

    using Newton iteration.
    """

    E = mean_anomaly

    for _ in range(20):

        f = E - eccentricity * np.sin(E) - mean_anomaly

        df = 1.0 - eccentricity * np.cos(E)

        E -= f / df

    return E


def satellite_position_ijk(time_s):


    period_s = orbit["period_hours"] * 3600.0

    mean_anomaly = (
        2.0
        * np.pi
        * (time_s % period_s)
        / period_s
    )

    e = orbit["eccentricity"]
    a = orbit["semi_major_axis_km"]

    eccentric_anomaly = solve_kepler(
        mean_anomaly,
        e,
    )


    x_orb = a * (
        np.cos(eccentric_anomaly) - e
    )

    y_orb = (
        a
        * np.sqrt(1.0 - e**2)
        * np.sin(eccentric_anomaly)
    )

    r_orb = np.array(
        [x_orb, y_orb, 0.0]
    )

    raan = np.deg2rad(
        orbit["raan_deg"]
    )

    inclination = np.deg2rad(
        orbit["inclination_deg"]
    )

    arg_perigee = np.deg2rad(
        orbit["arg_perigee_deg"]
    )

    # Orbital frame -> IJK
    r_ijk = (
        rotation_z(raan)
        @ rotation_x(inclination)
        @ rotation_z(arg_perigee)
        @ r_orb
    )

    return r_ijk




SUN_RA_DEG = orbit["raan_deg"]
SUN_DECLINATION_DEG = 0.0

sun_ra = np.deg2rad(SUN_RA_DEG)
sun_dec = np.deg2rad(SUN_DECLINATION_DEG)

SUN_IJK = np.array(
    [
        np.cos(sun_ra) * np.cos(sun_dec),
        np.sin(sun_ra) * np.cos(sun_dec),
        np.sin(sun_dec),
    ]
)




def environmental_fluxes(time_s):
    """
    Calculate:
        Q_sun      direct solar flux [W/m^2]
        Q_IR       Earth IR flux [W/m^2]
        Q_albedo   Earth-reflected solar flux [W/m^2]

    IR and albedo follow the Lambertian structure of
    Sasaki Eqs. (46) and (47).

    The spacecraft surface is nadir-facing, consistent
    with the paper's temperature demonstration.
    """


    r_sat_ijk = satellite_position_ijk(
        time_s
    )

    earth_rotation = (
        2.0
        * np.pi
        * time_s
        / SIDEREAL_DAY_S
    )

    transform = rotation_z(
        -earth_rotation
    )

    r_sat = transform @ r_sat_ijk
    sun_vector = transform @ SUN_IJK

    sun_vector /= np.linalg.norm(
        sun_vector
    )


    spacecraft_normal = (
        -r_sat / np.linalg.norm(r_sat)
    )




    sat_to_surface = (
        earth_points - r_sat
    )

    distance = np.linalg.norm(
        sat_to_surface,
        axis=1,
    )

    sat_to_surface_unit = (
        sat_to_surface
        / distance[:, None]
    )



    cos_theta_0 = (
        sat_to_surface_unit
        @ spacecraft_normal
    )


    cos_theta_1 = np.sum(
        earth_normals
        * (-sat_to_surface_unit),
        axis=1,
    )

    visible = (
        (cos_theta_0 > 0.0)
        &
        (cos_theta_1 > 0.0)
    )



    view_factor = (
        cos_theta_0
        * cos_theta_1
        * surface_area
        / (
            np.pi
            * distance**2
        )
    )




    Q_IR = np.sum(
        OLR_GRID[visible]
        * view_factor[visible]
    )




    cos_solar_zenith = (
        earth_normals
        @ sun_vector
    )

    illuminated = (
        visible
        &
        (cos_solar_zenith > 0.0)
    )

    Q_albedo = np.sum(
        G_ON
        * ALBEDO_GRID[illuminated]
        * cos_solar_zenith[illuminated]
        * view_factor[illuminated]
    )




    behind_earth = (
        np.dot(r_sat, sun_vector) < 0.0
    )

    perpendicular_distance = np.linalg.norm(
        r_sat
        - np.dot(r_sat, sun_vector)
        * sun_vector
    )

    in_eclipse = (
        behind_earth
        and
        perpendicular_distance
        < EARTH_RADIUS_KM
    )




    if in_eclipse:

        Q_sun = 0.0

    else:

        solar_incidence = max(
            np.dot(
                spacecraft_normal,
                sun_vector,
            ),
            0.0,
        )

        Q_sun = (
            G_ON
            * solar_incidence
        )


    return (
        Q_sun,
        Q_IR,
        Q_albedo,
        in_eclipse,
    )




period_s = (
    orbit["period_hours"]
    * 3600.0
)

total_time_s = (
    N_ORBITS
    * period_s
)

t_full = np.arange(
    0.0,
    total_time_s + DT_TH,
    DT_TH,
)

T_full = np.zeros_like(
    t_full,
    dtype=float,
)

Q_sun_full = np.zeros_like(
    t_full,
    dtype=float,
)

Q_IR_full = np.zeros_like(
    t_full,
    dtype=float,
)

Q_albedo_full = np.zeros_like(
    t_full,
    dtype=float,
)

eclipse_full = np.zeros_like(
    t_full,
    dtype=bool,
)

T_full[0] = INITIAL_TEMPERATURE_K


alpha = thermal["absorptivity"]

epsilon = thermal["emissivity"]

sigma = thermal["sigma_W_m2_K4"]

T_space = thermal[
    "deep_space_temperature_K"
]

C_node = thermal[
    "node_heat_capacity_J_K"
]


for i, time_s in enumerate(t_full):

    (
        Q_sun_full[i],
        Q_IR_full[i],
        Q_albedo_full[i],
        eclipse_full[i],
    ) = environmental_fluxes(
        time_s
    )

    if i == 0:
        continue




    Q_in = (
        alpha * Q_sun_full[i]
        + epsilon * Q_IR_full[i]
        + alpha * Q_albedo_full[i]
    )

    Q_out = (
        epsilon
        * sigma
        * (
            T_full[i - 1]**4
            - T_space**4
        )
    )

    dTdt = (
        Q_in - Q_out
    ) / C_node

    T_full[i] = (
        T_full[i - 1]
        + dTdt * DT_TH
    )




start_last_orbit = (
    (N_ORBITS - 1)
    * period_s
)

last_orbit_mask = (
    t_full >= start_last_orbit
)

t_th = (
    t_full[last_orbit_mask]
    - start_last_orbit
)

T_th = T_full[
    last_orbit_mask
]

Q_sun = Q_sun_full[
    last_orbit_mask
]

Q_IR = Q_IR_full[
    last_orbit_mask
]

Q_albedo = Q_albedo_full[
    last_orbit_mask
]

eclipse_mask = eclipse_full[
    last_orbit_mask
]

dt_th = DT_TH




if __name__ == "__main__":

    eclipse_hours = (
        np.sum(eclipse_mask)
        * dt_th
        / 3600.0
    )

    print(
        f"Orbital period: "
        f"{orbit['period_hours']:.6f} h"
    )

    print(
        f"Extraterrestrial solar flux G_on: "
        f"{G_ON:.2f} W/m^2"
    )

    print(
        f"Minimum temperature: "
        f"{T_th.min():.2f} K"
    )

    print(
        f"Maximum temperature: "
        f"{T_th.max():.2f} K"
    )

    print(
        f"Calculated eclipse duration: "
        f"{eclipse_hours:.2f} h/orbit"
    )




    plt.figure(figsize=(9, 5))

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

    plt.xlabel("Time (hours)")
    plt.ylabel("Incident Flux (W/m²)")

    plt.title(
        "HEO Thermal Environment "
        "(CERES-Derived Model)"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()




    plt.figure(figsize=(9, 5))

    plt.plot(
        t_th / 3600.0,
        T_th,
        label="Surface Temperature",
    )

    plt.xlabel("Time (hours)")
    plt.ylabel("Temperature (K)")

    plt.title(
        "HEO Temperature Over One Orbit "
        "(CERES-Derived Thermal Model)"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()