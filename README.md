# Optimal control of a quadcopter

This repository focuses on an implementation of optimal control for a quadcopter drone used in my Bachelor's thesis. Optimal control problem is solved using Model Predictive Control (MPC). MPC is approximated using a pseudospectral method.

## Tools
The MPC algorithm was implemented in Python using the MPOPT library (see [MPOPT](https://github.com/mpopt/mpopt)).

The functionality of the controller was demonstrated in the PyBullet simulation environment (see [PyBullet](https://github.com/bulletphysics/bullet3)) through two scenarios: a flight to a target point with an obstacle (see [here](https://github.com/ardno105/Optimalni_rizeni_dronu/blob/main/bakalarka_sim.py)) and following a predefined trajectory (see [here](https://github.com/ardno105/Optimalni_rizeni_dronu/blob/main/bakalarka_sim_elipsa_V2.py)).

The quadcopter used in this thesis is Crazyflie 2.0 which is implemented into the PyBullet environment using Gym-PyBullet-Drones library (see [Gym-PyBullet-Drones](https://github.com/utiasDSL/gym-pybullet-drones)).

## Files
The model of Crazyflie 2.0 is formulated and tested [here](https://github.com/ardno105/Optimalni_rizeni_dronu/blob/main/test_letu.py). 

The use of MPOPT for optimal control is demonstrated in a notebook [here](https://github.com/ardno105/Optimalni_rizeni_dronu/blob/main/bakalarka_mpopt.ipynb). 

The implementation of MPOPT for a point-flight with an obstacle is [here](https://github.com/ardno105/Optimalni_rizeni_dronu/blob/main/bakalarka_sim.py).

The implementation of MPOPT for trajectory tracking is [here](https://github.com/ardno105/Optimalni_rizeni_dronu/blob/main/bakalarka_sim_elipsa_V2.py).


> Written with [StackEdit](https://stackedit.io/).
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTkzMDM2MzczNV19
-->