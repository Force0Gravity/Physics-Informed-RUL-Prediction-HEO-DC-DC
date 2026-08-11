Physics-Informed RUL Prediction of a DC-DC Converter in HEO

This repository includes the Python code of a physics-informed framework for modeling degradation process and RUL prediction of a DC-DC converter working in a Highly Elliptical Orbit (HEO).

The developed model takes into account:

- HEO orbital information
- Thermal environment of a spacecraft
- Accumulated radiation dose 
- Thermal and radiation degradation indices
- Conventional neural network baselines
- Physics-Informed Neural Network (PINN)
- Monte Carlo sensitivity analysis
- Threshold-based RUL estimation

The workflow of the project consists of several stages:

- Definition of the HEO orbital, thermal and radiation environment based on source-based data.
- Calculation of heat fluxes from the Sun, Earth infrared background and albedo radiation and obtaining 
  the spacecraft temperature profile using the thermal model.
- Calculation of the cumulative radiation dose including the November storm event using the radiation model.
- Combination of the obtained thermal and radiation effects in a normalized degradation index.
- Prediction of the degradation index using Time-Only NN, Environmental NN, and Physics-Informed 
  Neural Network (PINN).
- Threshold-based RUL estimation using the predicted degradation.
