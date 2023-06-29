# Overview

This repository contains additional information and data relating to the contribution to the "1. Norddeutsche Wärmeforschungskonferenz" from the authors J. Freißmann, M. Fritz and I. Tuschy. It serves to allow for reproduction of the obtained results.

# The conference

You can find information regarding the conference on the respective [homepage](https://www.hawk.de/de/hochschule/fakultaeten-und-standorte/fakultaet-ressourcenmanagement/profil/nwf). There, you will also be able to find the book of abstracts and eventually the full paper.

# Description of contents

- [ ] Sonderborg network (Input/Output)
- [ ] TESPy (Setup 1, 2, 3 - .py & param)
- [ ] TESPy Characteristika

## Heat pump models

This folder contains two heat pump models and input data. The first model is a simple heat pump cycle containing only the basic components compressor, condenser, evaporator and valve. The second model employs the two most popular economizer topologies: a closed economizer, which is basically an internal heat exchanger and an open economizer with a flashtank. Additionally, you can find the parameter JSON files for the three setups used in the paper in this folder as well. Finally, the simulated part load characteristics of the three heat pumps are included.

## Unit commitment optimization

### Input data

This folder contains the necessary input data for the unit commitment optimization. Generally, this includes a JSON file of the constant parameters (no variation throughout the observed period) and a CSV file of the time dependent data. As all constant parameters stay the same for all three setups, the same JSON file is used. Each setup then has its own CSV file, as it includes the different part load characteristics.

### Output data

This folder contains the output data of the unit commitment optimization. For each setup, there are the unit commitment time series, as well as key parameters and unit cost. Additionally, the results are visualized in multiple plots, that allow an even more detailed analysis than provided in the paper.
