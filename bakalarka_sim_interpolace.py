"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

### Import knihoven pro MPC
from mpopt import mp
import casadi as ca

### Import pro výpočet interpolace
from scipy.interpolate import barycentric_interpolate

DEFAULT_DRONES = DroneModel("cf2p")     # Konfigurace dronu
DEFAULT_NUM_DRONES = 1                  # Počet dronů
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True                      # Vizualizace zapnuta
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True                     # Grafy výstpů
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True                # Překážky
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48   
DEFAULT_DURATION_SEC = 5
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

###################################################################################################################
### MODEL DRONU Z TEORIE #######
# Parametry
C_T = 1.28192e-8      # koeficient tahu rotoru
C_M = 5.964552e-3     # koeficient momentu rotoru
C_T2 = 3.16e-10      # koeficient tahu rotoru pro přepočet na rpm
C_M2 = 7.94e-12     # koeficient momentu rotoru
# C_T_test = 1e-5
f_min = 0
f_max = 0.15
arm_length = 0.0397     # délka ramene [m]

g = 9.81       # tíhové zrychlení [m/s²]
m = 0.027       # hmotnost dronu [kg]
# Ix = 1.657e-5  # moment setrvačnosti kolem osy x [kg·m²]
# Iy = 1.666e-5  # moment setrvačnosti kolem osy y [kg·m²]
# Iz = 2.926e-5  # moment setrvačnosti kolem osy z [kg·m²]
Ix = 1.4e-5  # moment setrvačnosti kolem osy x [kg·m²]
Iy = 1.4e-5  # moment setrvačnosti kolem osy y [kg·m²]
Iz = 2.17e-5  # moment setrvačnosti kolem osy z [kg·m²]

def dynamics1(x, u, t):
    # Definice stavů a vstupu
    # Stav: x = [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
    pos = x[0:3]      # poloha: [x, y, z]
    vel = x[3:6]      # rychlost: [vx, vy, vz]
    phi   = x[6]      # natočení
    theta = x[7]
    psi   = x[8]
    p     = x[9]      # úholvá rychlost
    q     = x[10]
    r     = x[11]

    # Celkový tah a momenty
    # F = u[0] + u[1] + u[2] + u[3]
    # tau_phi   = arm_length * (u[3] - u[1])
    # tau_theta = arm_length * (u[2] - u[0])
    # tau_psi   = (C_M / C_T) * (u[0] - u[1] + u[2] - u[3])

    u = np.sqrt(u/C_T2)

    # CF2X
    # F = C_T2*(u[0]**2 + u[1]**2 + u[2]**2 + u[3]**2)
    # tau_phi   = C_T2*(arm_length/np.sqrt(2)) * (u[0]**2 + u[1]**2 - u[2]**2 - u[3]**2)
    # tau_theta = C_T2*(arm_length/np.sqrt(2)) * (-u[0]**2 + u[1]**2 + u[2]**2 - u[3]**2)
    # tau_psi   = C_M2 * (u[0]**2 - u[1]**2 + u[2]**2 - u[3]**2)

    # CF2P
    F = C_T2*(u[0]**2 + u[1]**2 + u[2]**2 + u[3]**2)
    tau_phi   = C_T2*(arm_length/np.sqrt(2)) * (u[1]**2 - u[3]**2)
    tau_theta = C_T2*(arm_length/np.sqrt(2)) * (-u[0]**2 + u[2]**2)
    tau_psi   = C_M2 * (-u[0]**2 + u[1]**2 - u[2]**2 + u[3]**2)

    # Rotace dle Eulerových úhlů
    fx = (F/m) * (ca.cos(phi)*ca.sin(theta)*ca.cos(psi) + ca.sin(phi)*ca.sin(psi))
    fy = (F/m) * (ca.cos(phi)*ca.sin(theta)*ca.sin(psi) - ca.sin(phi)*ca.cos(psi))
    fz = (F/m) * (ca.cos(phi)*ca.cos(theta)) - g

    # Kinematika Eulerových úhlů
    phi_dot   = p + q * ca.sin(phi) * ca.tan(theta) + r * ca.cos(phi) * ca.tan(theta)
    theta_dot = q * ca.cos(phi) - r * ca.sin(phi)
    psi_dot   = q * ca.sin(phi) / ca.cos(theta) + r * ca.cos(phi) / ca.cos(theta)

    # Dynamika otáčení
    p_dot = (tau_phi - (Iy - Iz) * q * r) / Ix
    q_dot = (tau_theta - (Iz - Ix) * p * r) / Iy
    r_dot = (tau_psi - (Ix - Iy) * p * q) / Iz

    # Diferenciální rovnice
    x_dot = [vel[0],
                vel[1],
                vel[2],
                fx,
                fy,
                fz,
                phi_dot,
                theta_dot,
                psi_dot,
                p_dot,
                q_dot,
                r_dot]
    return x_dot
###################################################################################################################
###################################################################################################################
### Nastavení solveru pro výpočet MPC
# Definice OCP
ocp = mp.OCP(n_states=12, n_controls=4)

# Nastavení dynamiky pro OCP 
def get_dynamics1():
    dynamics0 = lambda x, u, t: dynamics1(x, u, t)

    return [dynamics0]
ocp.dynamics = get_dynamics1()

# Váhové matice Q a R 
# Q = np.diag([100, 100, 100, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
Q = np.diag([10,10,10, 1,1,1, 10,10,10, 1,1,1])
R = np.diag([1, 1, 1, 1])

### Nastavení hodnotící funkce ("let do bodu")
# x_ref = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
# u0 = np.array([0.073575, 0.073575, 0.073575, 0.073575])
# def running_cost1(x, u, t):
#     # Odchylka od trajektorie
#     x_err = x - x_ref
#     # Odchylka od stabilních vstupů
#     u_err = u - u0    
    
#     return (x_err.T @ Q @ x_err) + (u_err.T @ R @ u_err)
# ocp.running_costs[0] = running_cost1

# Omezení na cestu
# V kombinaci letem do bodu
def path_constrains0(x, u, t):
    x0 = 0
    y0 = 2
    z0 = 0.5
    return [
        (0.5+0.3)**2 - (x[0]-x0)*(x[0]-x0) - (x[1]-y0)*(x[1]-y0) - (x[2]-z0)*(x[2]-z0)
    ]

ocp.path_constraints[0] = path_constrains0

# Časový horizont (3 sekundy)
ocp.lbtf[0] = 3
ocp.ubtf[0] = 3

# Omezení na stav
# ocp.lbx[0] = np.array([[-np.inf, -np.inf, 0., -np.inf, -np.inf, -np.inf, -np.pi, -np.pi, -np.pi/2, -np.inf, -np.inf, -np.inf]])
# ocp.ubx[0] = np.array([[np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf, np.pi,  np.pi,  np.pi/2,  np.inf,  np.inf,  np.inf]])

# ocp.lbx[0] = np.array([[-np.inf,-np.inf,0.,   -np.inf, -np.inf, -np.inf, -30,-30,-30, -np.inf, -np.inf, -np.inf]])
# ocp.ubx[0] = np.array([[np.inf,np.inf,np.inf,  np.inf,  np.inf,  np.inf,  30, 30, 30,  np.inf,  np.inf,  np.inf]])

ocp.lbx[0] = np.array([[-np.inf,-np.inf,0.1,    -1,-1,-1,    -np.pi/4,-np.pi/4,-np.pi/2,  -5,-5,-5]])
ocp.ubx[0] = np.array([[np.inf,np.inf,np.inf,   1, 1, 1,     np.pi/4, np.pi/4, np.pi/2,   5, 5, 5]])

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    # INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])
    # INIT_XYZS = np.array([[0,0,1]])
    INIT_RPYS = np.array([[0, 0, 0]])

    #### Initialize a circular trajectory ######################
    PERIOD = 10
    # PERIOD = 3
    NUM_WP = control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))
    for i in range(NUM_WP):
        # TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], 0
        # TARGET_POS[i, :] = i/NUM_WP,i/NUM_WP,i/NUM_WP
        TARGET_POS[i,:] = 0,4,0.5
        # if i<control_freq_hz*3: 
        #     TARGET_POS[i, :] = 0,0,i/(control_freq_hz*3)
        # else: TARGET_POS[i, :] = 0,0,1
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])

    #### Debug trajectory ######################################
    #### Uncomment alt. target_pos in .computeControlFromState()
    # INIT_XYZS = np.array([[.3 * i, 0, .1] for i in range(num_drones)])
    # INIT_RPYS = np.array([[0, 0,  i * (np.pi/3)/num_drones] for i in range(num_drones)])
    # NUM_WP = control_freq_hz*15
    # TARGET_POS = np.zeros((NUM_WP,3))
    # for i in range(NUM_WP):
    #     if i < NUM_WP/6:
    #         TARGET_POS[i, :] = (i*6)/NUM_WP, 0, 0.5*(i*6)/NUM_WP
    #     elif i < 2 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-NUM_WP/6)*6)/NUM_WP, 0, 0.5 - 0.5*((i-NUM_WP/6)*6)/NUM_WP
    #     elif i < 3 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, ((i-2*NUM_WP/6)*6)/NUM_WP, 0.5*((i-2*NUM_WP/6)*6)/NUM_WP
    #     elif i < 4 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, 1 - ((i-3*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-3*NUM_WP/6)*6)/NUM_WP
    #     elif i < 5 * NUM_WP/6:
    #         TARGET_POS[i, :] = ((i-4*NUM_WP/6)*6)/NUM_WP, ((i-4*NUM_WP/6)*6)/NUM_WP, 0.5*((i-4*NUM_WP/6)*6)/NUM_WP
    #     elif i < 6 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-5*NUM_WP/6)*6)/NUM_WP
    # wp_counters = np.array([0 for i in range(num_drones)])

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )
    
    ### Nastavení počátečních podmínek pro MPC
    ocp.x00[0] = np.array([INIT_XYZS[0,0],INIT_XYZS[0,1],INIT_XYZS[0,2],0,0,0,INIT_RPYS[0,0],INIT_RPYS[0,1],INIT_RPYS[0,2],0,0,0])
    # ocp.x00[0] = np.array([0.,0.,0.5, 0.,0.,0., 0.,0.,0., 0.,0.,0.])
    # ocp.x00[0] = np.zeros([1,12])


    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    pid_action = np.zeros((num_drones,4))
    action = np.zeros((num_drones,4))

    ### Počáteční stav cíle
    x_ref = np.array([TARGET_POS[0,0],TARGET_POS[0,1],INIT_XYZS[0,2] ,0,0,0 ,0,0,0 ,0,0,0])
    # x_ref[0:2] = TARGET_POS[0, 0:2]
    # x_ref[2] = INIT_XYZS[0,2]

    # Omezení na vstup
    ocp.lbu[0] = 0.0
    # ocp.ubu[0] = C_T2*(env.MAX_RPM**2)
    # ocp.ubu[0] = 0.15
    ocp.ubu[0] = 0.13
    

    ### Stabilní vstup - hovering
    u_stable = C_T2*(env.HOVER_RPM**2)
    u0 = np.array([u_stable, u_stable, u_stable, u_stable])
    # u0 = np.array([0, 0, 0, 0])
    # u0 = np.array([0.073575, 0.073575, 0.073575, 0.073575])

    ### Počáteční výpočet MPC
    # ocp.running_costs[0] = lambda x, u, t: ((x-x_ref).T @ Q @ (x-x_ref) + (u-u0).T @ R @ (u-u0))
    # mpo, post = mp.solve(ocp, n_segments=2, poly_orders=10, scheme="LGR", plot=False)
    # opt = mp.mpopt(ocp, n_segments=1, poly_orders=10)
    # solution = opt.solve()
    # post = opt.process_results(solution, plot=False,scaling=False ,residual_dx=False)
    # data = post.get_data()
    # inputs = data[1][0]
    # action[0,:] = np.sqrt(np.abs(inputs/C_T2))

    # Nové body, kde chci hodnoty zinterpolovat
    # t_interp = np.linspace(0, 0.2083, 40)
    segment = 5
    t_interp = np.linspace(1/48, segment/48, segment)
    
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ/segment)):

        #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)
        

        #### Compute control for the current way point #############
        # for j in range(num_drones):
        #     pid_action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
        #                                                             state=obs[j],
        #                                                             target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]),
        #                                                             # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
        #                                                             target_rpy=INIT_RPYS[j, :]
        #                                                             )
            ### Výpočet MPC
            # mpo, post = mp.solve(ocp, n_segments=1, poly_orders=10, scheme="LGR", plot=False)
            # data = post.get_data()
            # inputs = data[1][0]
            # action[j,:] = inputs/C_T2
            # action[j, :] = np.array([[20000, 20000, 20000, 20000]])
        
        ### Výpočet MPC ##############################
        # ocp.running_costs[0] = lambda x, u, t: ((x-x_ref).T @ Q @ (x-x_ref) + (u-u0).T @ R @ (u-u0))
        # mpo, post = mp.solve(ocp, n_segments=1, poly_orders=10, scheme="LGR", plot=False)
        # data = post.get_data()
        # current_state = data[0][2]
        # inputs = data[1][0]
        # action[0,:] = np.sqrt(np.abs(inputs/C_T2))
        # index = index+1

        # x_ref[0:3] = TARGET_POS[i+1]
        # wp_counters[0] = wp_counters[0] + 1 if wp_counters[0] < (NUM_WP-1) else 0
        # # x_ref[0:2] = TARGET_POS[wp_counters[0],0:2]
        # x_ref = np.array([TARGET_POS[wp_counters[0],0],TARGET_POS[wp_counters[0],1],TARGET_POS[wp_counters[0],2], 0,0,0 ,0,0,0 ,0,0,0])
        # vx = (TARGET_POS[i,0]-current_state[0])/2
        # vy = (TARGET_POS[i,1]-current_state[1])/2
        # vz = (TARGET_POS[i,2]-current_state[2])/2
        # x_ref = np.array([TARGET_POS[i,0], TARGET_POS[i,1],TARGET_POS[i,2], vx,vy,vz ,0,0,0 ,0,0,0])
        # x_ref = np.array([0,0,1, 0,0,0, 0,0,1, 0,0,0])

        #### Go to the next way point and loop #####################
        # for j in range(num_drones):
        #     wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0
        #     # x_ref[0:3] = TARGET_POS[wp_counters[j]]
        #     # x_ref[2] = INIT_XYZS[0,2]
        #     x_ref = np.array([TARGET_POS[wp_counters[j],0], TARGET_POS[wp_counters[j],1],TARGET_POS[wp_counters[j],2], 0,0,0 ,0,0,0 ,0,0,0])

        ocp.running_costs[0] = lambda x, u, t: ((x-x_ref).T @ Q @ (x-x_ref) + (u-u0).T @ R @ (u-u0))
        mpo, post = mp.solve(ocp, n_segments=1, poly_orders=10, scheme="LGR", plot=False)
        data = post.get_data()
        current_state = data[0][2]
        t = data[2].T
        t = t[0]
        inputs = data[1].T
        input1 = inputs[0]
        input2 = inputs[1]
        input3 = inputs[2]
        input4 = inputs[3]
        # Barycentrická interpolace
        u1 = barycentric_interpolate(t, input1, t_interp)
        u2 = barycentric_interpolate(t, input2, t_interp)
        u3 = barycentric_interpolate(t, input3, t_interp)
        u4 = barycentric_interpolate(t, input4, t_interp)
        

        # action[0,:] = np.sqrt(np.abs(inputs/C_T2))

        wp_counters[0] = wp_counters[0] + 1 if wp_counters[0] < (NUM_WP-1) else 0
        x_ref = np.array([TARGET_POS[wp_counters[0],0],TARGET_POS[wp_counters[0],1],TARGET_POS[wp_counters[0],2], 0,0,0 ,0,0,0 ,0,0,0])

        print("cilovy stav: ", x_ref)
        print("vstupy: ", action[0,:])

        #### Step the simulation ###################################
        for j in range(segment):
            u_interp = np.array([u1[j], u2[j], u3[j], u4[j]])
            action[0,:] = np.sqrt(np.abs(u_interp/C_T2))
            obs, reward, terminated, truncated, info = env.step(action)

            #### Log the simulation ####################################
            logger.log(drone=0,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[0],
                       control=np.hstack([TARGET_POS[wp_counters[0], 0:2], INIT_XYZS[0, 2], INIT_RPYS[0, :], np.zeros(6)])
                       # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                       )

            #### Printout ##############################################
            env.render()

            #### Sync the simulation ###################################
            if gui:
                sync(i, START, env.CTRL_TIMESTEP)

        ### Změna stavu...nové x0 pro další výpočet MPC
        state = env._getDroneStateVector(0)
        new_xyz = state[0:3]
        new_vxvyvz = state[10:13]
        new_phithetapsi = state[7:10]
        # new_phithetapsi = p.getEulerFromQuaternion(state[3:7])
        new_pqr = state[13:16]
        ocp.x00[0][0:3] = new_xyz
        ocp.x00[0][3:6] = new_vxvyvz
        ocp.x00[0][6:9] = new_phithetapsi
        ocp.x00[0][9:12] = new_pqr
        # ocp.x00[0] = current_state
        # ocp.u00[0] = action[0,:]
        print("akt. stav: ", new_xyz)
        # env.state
        

        # #### Log the simulation ####################################
        # for j in range(num_drones):
        #     logger.log(drone=j,
        #                timestamp=i/env.CTRL_FREQ,
        #                state=obs[j],
        #                control=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
        #                # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
        #                )

        # #### Printout ##############################################
        # env.render()

        # #### Sync the simulation ###################################
        # if gui:
        #     sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    print("hotovo")
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

    # print("hotovo")

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
