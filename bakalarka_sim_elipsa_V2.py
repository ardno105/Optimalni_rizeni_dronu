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

### 3D vykreslení
from mpl_toolkits.mplot3d import Axes3D

DEFAULT_DRONES = DroneModel("cf2p")     # Konfigurace dronu
DEFAULT_NUM_DRONES = 1                  # Počet dronů
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = False                      # Vizualizace zapnuta
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True                     # Grafy výstpů
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True                # Překážky
# DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_SIMULATION_FREQ_HZ = 960
DEFAULT_CONTROL_FREQ_HZ = 120   
# DEFAULT_CONTROL_FREQ_HZ = 240   
DEFAULT_DURATION_SEC = 10
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

    u = np.sqrt(u/C_T2)

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
# # Pro elipsu
Q = np.diag([100,100,100,   1,1,1,   1,1,1,   1,1,1])
Qf = np.diag([100,100,100,   100,100,100,   100,100,100,   100,100,100])
# Qf = np.diag([100, 100, 100, 1,1,1, 1,1,1, 1,1,1])
# Pro let do bodu
# Q = np.diag([10,10,10, 1,1,1, 10,10,10, 1,1,1])
R = np.diag([1, 1, 1, 1])

# Omezení na cestu
# V kombinaci letem do bodu
def path_constrains0(x, u, t):
    x0 = 0
    y0 = 2
    z0 = 0.5
    return [
        (0.5+0.3)**2 - (x[0]-x0)*(x[0]-x0) - (x[1]-y0)*(x[1]-y0) - (x[2]-z0)*(x[2]-z0)
    ]

# ocp.path_constraints[0] = path_constrains0

# Časový horizont (3 sekundy)
ocp.lbtf[0] = 5
ocp.ubtf[0] = 5

# Omezení na stav
# ocp.lbx[0] = np.array([[-np.inf, -np.inf, 0.1, -2, -2, -2, -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf]])
# ocp.ubx[0] = np.array([[np.inf,  np.inf,  5,  2,  2,  2, np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf]])

# Let po spirale
ocp.lbx[0] = np.array([[-np.inf,-np.inf,0.1,   -np.inf, -np.inf, -np.inf, -np.pi,-np.pi,-np.pi/2, -np.inf, -np.inf, -np.inf]])
ocp.ubx[0] = np.array([[np.inf,np.inf,np.inf,  np.inf,  np.inf,  np.inf,  np.pi, np.pi, np.pi/2,  np.inf,  np.inf,  np.inf]])

# Pro let  do bodu
# ocp.lbx[0] = np.array([[-np.inf,-np.inf,0.1,    -1,-1,-1,    -np.pi/2,-np.pi/2,-np.pi/2,  -np.inf,-np.inf,-np.inf]])
# ocp.ubx[0] = np.array([[np.inf,np.inf,np.inf,   1, 1, 1,     np.pi/2, np.pi/2, np.pi/2,    np.inf, np.inf, np.inf]])

def printPointFlight(logger, init_xyz, x_ref,J , duration_sec, segment):
    plt.rcParams.update({
        'text.usetex': False,        # aktivuje LaTeX pro text
        'font.size': 20,
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'legend.fontsize': 12,
    })

    fig, ax = plt.subplots(figsize=(6,4))
    fig.subplots_adjust(
        left=0.07,   # okraj vlevo
        right=0.985,  # okraj vpravo
        bottom=0.1, # spodní okraj
        top=0.987,    # horní okraj
        wspace=0.3, # vodorovná mezera mezi subploty
        hspace=0.4  # svislá mezera mezi subploty
    )
    ax.plot(np.linspace(0, duration_sec, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ)), logger.states[0][0], label='x')
    ax.plot(np.linspace(0, duration_sec, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ)), logger.states[0][1], label='y')
    ax.plot(np.linspace(0, duration_sec, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ)), logger.states[0][2], label='z')
    ax.plot(np.linspace(0, duration_sec, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ)), np.ones([int(duration_sec*DEFAULT_CONTROL_FREQ_HZ),1])*x_ref[0],color='black', linestyle='--', label='Reference')
    ax.plot(np.linspace(0, duration_sec, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ)), np.ones([int(duration_sec*DEFAULT_CONTROL_FREQ_HZ),1])*x_ref[1],color='black', linestyle='--')
    ax.plot(np.linspace(0, duration_sec, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ)), np.ones([int(duration_sec*DEFAULT_CONTROL_FREQ_HZ),1])*x_ref[2],color='black', linestyle='--')
    ax.grid(True)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m]')
    ax.legend()

    fig = plt.figure(figsize=(6,4))
    fig.subplots_adjust(
        left=0.086,   # okraj vlevo
        right=0.987,  # okraj vpravo
        bottom=0.088, # spodní okraj
        top=0.98,    # horní okraj
        wspace=0.204, # vodorovná mezera mezi subploty
        hspace=0.5  # svislá mezera mezi subploty
    )
    gs = fig.add_gridspec(3, hspace=0.5)
    axs = gs.subplots( sharex=True)
    for i in range(3):  
        axs[0].plot(np.linspace(0, duration_sec, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ)), logger.states[0][3+i])
        axs[1].plot(np.linspace(0, duration_sec, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ)), logger.states[0][6+i])
        axs[2].plot(np.linspace(0, duration_sec, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ)), logger.states[0][9+i])

        axs[0].grid(True)
        axs[1].grid(True)
        axs[2].grid(True)

        axs[0].set_ylabel("Velocity [m/s]", fontsize = 16)
        axs[0].legend(["u", "v", "w"])

        axs[1].set_ylabel("Orientation [rad]", fontsize = 16)
        axs[1].legend([r"$\phi$", r"$\theta$", r"$\psi$"])

        axs[2].set_xlabel("Time [s]")
        axs[2].set_ylabel("Angular velocity [rad/s]", fontsize = 16)
        axs[2].legend(["p", "q", "r"])  
    plt.show()

    # 3D vykresleni
    # Parametry koule
    r = 0.5  
    center = np.array([0, 2, 0.5]) 
    # Sférické souřadnice
    u = np.linspace(0, 2 * np.pi, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ))
    v = np.linspace(0, np.pi, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ))
    # Parametrické rovnice koule se středem
    x = center[0] + r * np.outer(np.cos(u), np.sin(v))
    y = center[1] + r * np.outer(np.sin(u), np.sin(v))
    z = center[2] + r * np.outer(np.ones_like(u), np.cos(v))
    # Vykreslení
    fig = plt.figure()
    plt.rcParams.update({
        'text.usetex': False,        # aktivuje LaTeX pro text
        'font.size': 14,
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'legend.fontsize': 12,
    })
    fig.subplots_adjust(
        left=0.12,   # okraj vlevo
        right=0.9,  # okraj vpravo
        bottom=0, # spodní okraj
        top=1,    # horní okraj
        wspace=0.2, # vodorovná mezera mezi subploty
        hspace=0.2  # svislá mezera mezi subploty
    )
    ax = fig.add_subplot(111, projection='3d')

    # Průhledná koule
    ax.plot_surface(x, y, z, color='blue', alpha=0.3, edgecolor='none')
    ax.plot(logger.states[0][0], logger.states[0][1], logger.states[0][2], 'r', linewidth=2, label='Position')
    ax.plot(init_xyz[0,0],init_xyz[0,1],init_xyz[0,2], color='black', marker='o', label='Start Point')
    ax.plot(logger.states[0][0][-1],logger.states[0][1][-1],logger.states[0][2][-1], color='black', marker='x', label='End Point')
    ax.set_xlabel('x [m]', fontsize=18, labelpad=15)
    ax.set_ylabel('y [m]', fontsize=18, labelpad=15)
    ax.set_zlabel('z [m]', fontsize=18, labelpad=15)
    ax.legend()
    plt.show()

    # Vykresleni inputu
    fig = plt.figure()
    plt.rcParams.update({
        'text.usetex': False,        # aktivuje LaTeX pro text
        'font.size': 20,
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'legend.fontsize': 12,
    })
    fig.subplots_adjust(
        left=0.07,   # okraj vlevo
        right=0.985,  # okraj vpravo
        bottom=0.1, # spodní okraj
        top=0.987,    # horní okraj
        wspace=0.3, # vodorovná mezera mezi subploty
        hspace=0.4  # svislá mezera mezi subploty
    )
    for i in range(4):
        plt.plot(np.linspace(0, duration_sec, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ)), logger.states[0][12+i])
    plt.grid(True)
    plt.ylabel("Input [RPM]")
    plt.xlabel("Time [s]")
    plt.legend([r"$u_1$", r"$u_2$", r"$u_3$", r"$u_4$"])

    # Vykresleni funkce J
    fig = plt.figure()
    plt.plot(np.linspace(0, duration_sec, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ/segment)), J)
    plt.ylabel("J")
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.show()

def printTrajectoryFlight(logger, init_xyz, trajectory, J , duration_sec, segment, elipse_step):
    t_sim = int(duration_sec*DEFAULT_CONTROL_FREQ_HZ)
    t = np.linspace(0, duration_sec, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ))
    x = np.linspace(0, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ/segment), int(duration_sec*DEFAULT_CONTROL_FREQ_HZ))
    # Vykresleni stavu
    plt.rcParams.update({
        'text.usetex': False,        # aktivuje LaTeX pro text
        'font.size': 20,
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'legend.fontsize': 12,
    })

    fig, ax = plt.subplots(figsize=(6,4))
    fig.subplots_adjust(
        left=0.07,   # okraj vlevo
        right=0.985,  # okraj vpravo
        bottom=0.1, # spodní okraj
        top=0.987,    # horní okraj
        wspace=0.3, # vodorovná mezera mezi subploty
        hspace=0.4  # svislá mezera mezi subploty
    )
    ax.plot(np.linspace(0, duration_sec, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ)), logger.states[0][0], label='x')
    ax.plot(np.linspace(0, duration_sec, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ)), logger.states[0][1], label='y')
    ax.plot(np.linspace(0, duration_sec, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ)), logger.states[0][2], label='z')
    ax.plot(t, np.cos((x)/elipse_step),color='black', linestyle='--', label='Reference')
    ax.plot(t, np.sin((x)/elipse_step),color='black', linestyle='--')
    ax.plot(t, np.linspace(1,2,1200),color='black', linestyle='--')
    ax.grid(True)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m]')
    ax.legend()
    plt.show()

    fig = plt.figure(figsize=(6,4))
    fig.subplots_adjust(
        left=0.086,   # okraj vlevo
        right=0.987,  # okraj vpravo
        bottom=0.088, # spodní okraj
        top=0.98,    # horní okraj
        wspace=0.204, # vodorovná mezera mezi subploty
        hspace=0.5  # svislá mezera mezi subploty
    )
    gs = fig.add_gridspec(3, hspace=0.5)
    axs = gs.subplots( sharex=True)
    for i in range(3):  
        axs[0].plot(np.linspace(0, duration_sec, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ)), logger.states[0][3+i])
        axs[1].plot(np.linspace(0, duration_sec, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ)), logger.states[0][6+i])
        axs[2].plot(np.linspace(0, duration_sec, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ)), logger.states[0][9+i])

        axs[0].grid(True)
        axs[1].grid(True)
        axs[2].grid(True)

        axs[0].set_ylabel("Velocity [m/s]", fontsize = 16)
        axs[0].legend(["u", "v", "w"])

        axs[1].set_ylabel("Orientation [rad]", fontsize = 16)
        axs[1].legend([r"$\phi$", r"$\theta$", r"$\psi$"])

        axs[2].set_xlabel("Time [s]")
        axs[2].set_ylabel("Angular velocity [rad/s]", fontsize = 16)
        axs[2].legend(["p", "q", "r"])  
    plt.show()

    # 3D vykresleni
    # Parametry koule
    # Vykreslení
    fig = plt.figure()
    plt.rcParams.update({
        'text.usetex': False,        # aktivuje LaTeX pro text
        'font.size': 14,
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'legend.fontsize': 12,
    })
    fig.subplots_adjust(
        left=0.12,   # okraj vlevo
        right=0.9,  # okraj vpravo
        bottom=0, # spodní okraj
        top=1,    # horní okraj
        wspace=0.2, # vodorovná mezera mezi subploty
        hspace=0.2  # svislá mezera mezi subploty
    )
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(logger.states[0][0], logger.states[0][1], logger.states[0][2], 'r', linewidth=2, label='Position')
    ax.plot(init_xyz[0,0],init_xyz[0,1],init_xyz[0,2], color='black', marker='o')
    # ax.plot(np.cos((x-10)/elipse_step),np.sin((x-10)/elipse_step),np.linspace(1,2,1200), color='black', linestyle='--', label='Reference')
    ax.plot(np.cos((x)/elipse_step),np.sin((x)/elipse_step),np.linspace(1,2,1200), color='black', linestyle='--', label='Reference')
    ax.set_xlabel('x [m]', fontsize=18, labelpad=15)
    ax.set_ylabel('y [m]', fontsize=18, labelpad=15)
    ax.set_zlabel('z [m]', fontsize=18, labelpad=15)
    ax.legend()
    plt.show()

    fig.subplots_adjust(
        left=0.07,   # okraj vlevo
        right=0.985,  # okraj vpravo
        bottom=0.1, # spodní okraj
        top=0.987,    # horní okraj
        wspace=0.3, # vodorovná mezera mezi subploty
        hspace=0.4  # svislá mezera mezi subploty
    )
    for i in range(4):
        plt.plot(np.linspace(0, duration_sec, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ)), logger.states[0][12+i])
    plt.grid(True)
    plt.ylabel("Input [RPM]")
    plt.xlabel("Time [s]")
    plt.legend([r"$u_1$", r"$u_2$", r"$u_3$", r"$u_4$"])

    # Vykresleni funkce J
    fig = plt.figure()
    plt.plot(np.linspace(0, duration_sec, int(duration_sec*DEFAULT_CONTROL_FREQ_HZ/segment)), J)
    plt.ylabel("J")
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.show()

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
    # INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    # INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])
    
    ## Pocatek pro let do bodu
    # INIT_XYZS = np.array([[0,0,0.1]])

    ## Pocatek pro let po spirale
    INIT_XYZS = np.array([[0.5,0,1]])
    INIT_RPYS = np.array([[0, 0, 0]])

    #### Initialize a circular trajectory ######################
    PERIOD = 10
    # PERIOD = 3
    NUM_WP = control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))
    for i in range(NUM_WP):
        TARGET_POS[i,:] = 0,4,0.5
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])

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
    # x_ref = np.array([TARGET_POS[0,0],TARGET_POS[0,1],INIT_XYZS[0,2] ,0,0,0 ,0,0,0 ,0,0,0])
    x_ref = np.array([INIT_XYZS[0,0],INIT_XYZS[0,1],INIT_XYZS[0,2], 0,0,0, 0,0,0, 0,0,0])
    x_ref10 = np.array([INIT_XYZS[0,0],INIT_XYZS[0,1],INIT_XYZS[0,2], 0,0,0, 0,0,0, 0,0,0])
    trajectory = np.zeros([int(duration_sec*env.CTRL_FREQ), 3])

    # Omezení na vstup
    ocp.lbu[0] = 0.0
    # ocp.ubu[0] = C_T2*(env.MAX_RPM**2)
    # ocp.ubu[0] = 0.15
    ocp.ubu[0] = 0.13
    

    ### Stabilní vstup - hovering
    u_stable = C_T2*(env.HOVER_RPM**2)
    u0 = np.array([u_stable, u_stable, u_stable, u_stable])
    ocp.u00[0] = u0
    # u0 = np.array([0, 0, 0, 0])

    segment = 10
    time_step = duration_sec/env.PYB_FREQ
    t_interp = np.linspace(time_step, segment*time_step, segment)

    J = np.zeros([int(duration_sec*env.CTRL_FREQ/segment), 1])
    elipse_step = 20

    t_sim = int(duration_sec*env.CTRL_FREQ/segment)
    
    START = time.time()



    for i in range(0, int(duration_sec*env.CTRL_FREQ/segment)):
        # ocp.running_costs[0] = lambda x, u, t: ((x-np.array([np.cos((3/5)*(t+i/12)), np.sin((3/5)*(t+i/12)), 1+i/120+t/5, -np.sin((3/5)*(t+i/12))+3/5,np.cos((3/5)*(t+i/12))+3/5,1/5, 0,0,0, 0,0,0])).T @ Q @ (x-np.array([np.cos((3/5)*(t+i/12)), np.sin((3/5)*(t+i/12)), 1+i/120+t/5, -np.sin((3/5)*(t+i/12))+3/5,np.cos((3/5)*(t+i/12))+3/5,1/5, 0,0,0, 0,0,0])) + (u-u0).T @ R @ (u-u0))
        ocp.running_costs[0] = lambda x, u, t: ((x-np.array([np.cos((3/5)*(t+i/12)), np.sin((3/5)*(t+i/12)), 1+i/120+t/5, 0,0,0, 0,0,0, 0,0,0])).T @ Q @ (x-np.array([np.cos((3/5)*(t+i/12)), np.sin((3/5)*(t+i/12)), 1+i/120+t/5, 0,0,0, 0,0,0, 0,0,0])) + (u-u0).T @ R @ (u-u0))
        # ocp.running_costs[0] = lambda x, u, t: ((x-x_ref).T @ Q @ (x-x_ref) + (u-u0).T @ R @ (u-u0))
        # Pro let po spirale
        # ocp.terminal_costs[0] = lambda xf, tf, x0, t0: ((xf-np.array([np.cos((3/5)*(tf+i/12)), np.sin((3/5)*(tf+i/12)), 1+i/120+tf/5, -np.sin((3/5)*(tf+i/12))+3/5,np.cos((3/5)*(tf+i/12))+3/5,1/5, 0,0,0, 0,0,0])).T @ Qf @ (xf-np.array([np.cos((3/5)*(tf+i/12)), np.sin((3/5)*(tf+i/12)), 1+i/120+tf/5, -np.sin((3/5)*(tf+i/12))+3/5,np.cos((3/5)*(tf+i/12))+3/5,1/5, 0,0,0, 0,0,0])))
        # ocp.terminal_costs[0] = lambda xf, tf, x0, t0: ((xf-np.array([np.cos((3/5)*(tf+i/12)), np.sin((3/5)*(tf+i/12)), 1+i/120+tf/5, 0,0,0, 0,0,0, 0,0,0])).T @ Qf @ (xf-np.array([np.cos((3/5)*(tf+i/12)), np.sin((3/5)*(tf+i/12)), 1+i/120+tf/5, 0,0,0, 0,0,0, 0,0,0])))
        mpo, post = mp.solve(ocp, n_segments=1, poly_orders=10, scheme="LGR", plot=False)
        data = post.get_data()
        sol = post.solution
        J[i] = sol['f']
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
        
        wp_counters[0] = wp_counters[0] + 1 if wp_counters[0] < (NUM_WP-1) else 0

        # # Let do bodu
        # x_ref = np.array([TARGET_POS[wp_counters[0],0],TARGET_POS[wp_counters[0],1],TARGET_POS[wp_counters[0],2], 0,0,0 ,0,0,0 ,0,0,0])
        # trajectory[i] = [x_ref[0], x_ref[1], x_ref[2]]
        
        # Let po spirale
        x_ref = np.array([np.cos(i/elipse_step),np.sin(i/elipse_step),INIT_XYZS[0,2]+i/120, -np.sin(i/elipse_step)*(1/elipse_step),np.cos(i/elipse_step)*(1/elipse_step),0 ,0,0,0 ,0,0,0])
        x_ref10 = np.array([np.cos((i+10)/elipse_step),np.sin((i+10)/elipse_step),INIT_XYZS[0,2]+(i+10)/120, -np.sin((i+10)/elipse_step)*(1/elipse_step),np.cos((i+10)/elipse_step)*(1/elipse_step),0 ,0,0,0 ,0,0,0])
        trajectory[i] = [x_ref[0], x_ref[1], x_ref[2]]

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
        new_pqr = state[13:16]
        ocp.x00[0][0:3] = new_xyz
        ocp.x00[0][3:6] = new_vxvyvz
        ocp.x00[0][6:9] = new_phithetapsi
        ocp.x00[0][9:12] = new_pqr
        print("akt. stav: ", new_xyz)

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)
        

    #### Close the environment #################################
    print("hotovo")
    env.close()

    # #### Save the simulation results ###########################
    # logger.save()
    # logger.save_as_csv("MPC_traj") # Optional CSV save

    # #### Plot the simulation results ###########################
    # if plot:
    #     logger.plot()

    #### Vykreslí simulaci letu do bodu
    # printPointFlight(logger, INIT_XYZS, x_ref, J, duration_sec, segment)

    #### Vykreslí simulaci letu po trajektorii
    trajectory = trajectory.T
    printTrajectoryFlight(logger,INIT_XYZS, trajectory, J, duration_sec, segment, elipse_step)
    
    
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
