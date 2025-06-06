import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Parametry
# =============================================================================
# C_T = 1.28192e-8      # koeficient tahu rotoru
# C_M = 5.964552e-3     # koeficient momentu rotoru
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
Ts = 1e-3    # perioda vzorkování [s]

# Limity tahu
f_max = 0.15 * 4   # maximální tah všech 4 rotorů (u jednoho f_i_max = 0.15 N)
f_min = 0.0

# Limity momentů
tau_max = f_max * arm_length
tau_min = -tau_max

pwm_max = 65535
rpm_max = 0.2685*pwm_max + 4070.3            # maximální otáčky [rpm]
# =============================================================================
# Definice stavu, vstupu a dynamiky pomocí CasADi
# =============================================================================
# Stavový vektor: x = [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
n_states = 12
n_controls = 4

x = ca.SX.sym('x', n_states)
u = ca.SX.sym('u', n_controls)  # vstup: jednotlivé tahy rotorů [f1, f2, f3, f4]

# Definice stavových veličin
pos = x[0:3]      # poloha: [x, y, z]
vel = x[3:6]      # rychlost: [vx, vy, vz]
phi   = x[6]      # roll
theta = x[7]      # pitch
psi   = x[8]      # yaw
p     = x[9]      # úhlová rychlost kolem x
q     = x[10]     # úhlová rychlost kolem y
r     = x[11]     # úhlová rychlost kolem z

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

# Sestavení diferenciálních rovnic:
x_dot = ca.vertcat(vel[0],
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
                   r_dot)

# Casadi funkce pro dynamiku
f_dyn = ca.Function('f_dyn', [x, u], [x_dot])

# Nastavení DAE pro integrátor (s použitím schématu RK4)
dae = {'x': x, 'p': u, 'ode': x_dot}
opts = {'tf': Ts}  # integrace přes jeden vzorkovací krok
integrator = ca.integrator('integrator', 'rk', dae, opts)

# =============================================================================
# Simulace trajektorie při libovolném konstantním vstupu
# =============================================================================
sim_time = 1.0               # délka simulace [s]
N_sim = int(sim_time / Ts)   # počet simulačních kroků

# Počáteční stav: dron v klidu, na zemi (poloha 0, rychlost 0, úhly 0)
x0 = np.zeros(n_states)

# Vstup: 
u_max_input = 0.15  
# u_sim = np.array([u_max_input, u_max_input, u_max_input, u_max_input])    # konstantní maximální tah u každého rotoru (tj. 0.15 N)
u_sim = np.array([rpm_max, 0, rpm_max, 0])      # libovolný vstup

# Uložení historie stavu
state_history = np.zeros((N_sim + 1, n_states))
time_history = np.zeros(N_sim + 1)
state_history[0, :] = x0

x_current = x0.copy()
for k in range(N_sim):
    res = integrator(x0=x_current, p=u_sim)
    x_current = res['xf'].full().flatten()
    state_history[k + 1, :] = x_current
    time_history[k + 1] = time_history[k] + Ts

# =============================================================================
# Vykreslení výsledků
# =============================================================================
# Vývoj x-ové souřadnice v čase
plt.figure()
plt.plot(time_history, state_history[:, 0], 'b', linewidth=2)
plt.xlabel('Čas [s]')
plt.ylabel('Výška [m]')
plt.title('Trajektorie dronu v ose x')
plt.grid(True)
plt.show()

# Vývoj y-ové souřadnice v čase
plt.figure()
plt.plot(time_history, state_history[:, 1], 'b', linewidth=2)
plt.xlabel('Čas [s]')
plt.ylabel('Výška [m]')
plt.title('Trajektorie dronu v ose y')
plt.grid(True)
plt.show()

# Vývoj výšky (z-ová souřadnice) v čase
plt.figure()
plt.plot(time_history, state_history[:, 2], 'b', linewidth=2)
plt.xlabel('Čas [s]')
plt.ylabel('Výška [m]')
plt.title('Trajektorie dronu v ose z')
plt.grid(True)
plt.show()

# 3D trajektorie (poloha v prostoru)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(state_history[:, 0], state_history[:, 1], state_history[:, 2], 'r', linewidth=2)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('3D trajektorie dronu')
plt.show()

plt.subplot(1,3,1)
plt.plot(state_history[:, 6])
plt.title("Phi")
plt.xlabel("Time, s")
plt.subplot(1,3,2)
plt.plot(state_history[:, 7])
plt.title("Theta")
plt.xlabel("Time, s")
plt.subplot(1,3,3)
plt.plot(state_history[:, 8])
plt.title("Psi")
plt.xlabel("Time, s")
