

import numpy as np



orbit = {

    "period_hours": 11.963582,


    "semi_major_axis_km": 26547.5,
    "eccentricity": 0.688,
    "inclination_deg": 63.4,
    "raan_deg": 153.0,
    "arg_perigee_deg": 270.0,
}



radiation = {


    "month_name": "November",
    "days_in_model_month": 30.0,


    "monthly_dose_rad": 7.03,

    "storm_day": 10.0,

    "storm_factor_min": 10.0,
    "storm_factor_max": 100.0,


    "reference_storm_factor": 10.0,


    "shielding_mil_be": 111.0,
    "shielding_mil_al_equivalent": 76.0,
}




thermal = {

    "solar_constant_W_m2": 1361.1,


    "absorptivity": 0.9,
    "emissivity": 0.9,

    "sigma_W_m2_K4": 5.6693e-8,


    "deep_space_temperature_K": 2.73,


    "node_heat_capacity_J_K": 13440.0,


    "day_of_year": 315,
}




CERES_OLR_COEFF = np.array([
    [846.5127,  -11.71082,  -5.007011],
    [  9.35777, -38.16887, -11.64853],
    [-84.64440,  -4.148072, -1.399822],
    [ 12.61828, -24.18759,  -6.167508],
    [-22.10449,   6.610096,  2.565666],
], dtype=float)




CERES_ALBEDO_COEFF = np.array([
    [ 1.135866,    0.02805531,   0.004863929],
    [-0.02568558,  0.1560044,    0.02306313 ],
    [ 0.4129173,   0.03064179,   0.01157133 ],
    [-0.04970988,  0.03961453,   0.02163336 ],
    [ 0.1107932,   0.002905725, -0.01802534 ],
], dtype=float)


ceres_uncertainty = {
    "OLR_RMSE_W_m2": 31.38267,
    "albedo_RMSE": 0.1147665,
}


