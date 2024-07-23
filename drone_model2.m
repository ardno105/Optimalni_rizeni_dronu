function model = drone_model()
%DRONE_MODEL2 Summary of this function goes here
%   Detailed explanation goes here
addpath('C:\Users\Ondra\VS\bakalarka\CasADi');
import casadi.*

%% Parametry
% kT = 3.16e-10; % rotor thrust coefficient
% ktau = 7.94e-12; % rotor torque coefficient
C_T = 1.28192e-8; % rotor thrust coefficient
C_M = 5.964552e-3; % rotor torque coefficient

rad_max = (2*pi*64*200)/60; % maximal angular rate of rotor [rad/s]
rad_min = (2*pi*64*18)/60; % minimal angular rate of rotor [rad/s]
% rpm_max = rad_max*9.5492968;    % [rpm]
% rpm_min = rad_min*9.5492968;    % [rpm]
arm_length = 0.0397; % arm length

g = 9.81; % gravitational acceleration
m = 0.03; % mass of quadrotor
m_lh = 0.032; % mass of quadrotor with lighthouse deck

% Ix = 2.3951e-5; % inertia of quadrotor around x-axis     
% Iy = 6.410179e-6; % inertia of quadrotor around y-axis
% Iz = 9.860228e-6; % inertia of quadrotor around z-axis
Ix = 1.395e-5; % inertia of quadrotor around x-axis     
Iy = 1.436e-6; % inertia of quadrotor around y-axis
Iz = 2.173e-6; % inertia of quadrotor around z-axis


Ts = 1e-3; % sampling period

% limit thrust
f_max = 0.15 * 4; % maximum collective thrust (single thrust f_i_max = 0.15)
f_min = 0.0;

% limit torque
tau_max = f_max * arm_length - f_min * arm_length;
tau_min = -tau_max;

% aerodynamic effect
K_aero = 1e-7 * [-10.2506, -0.3177, -0.4332;
                 -0.3177, -10.2506, -0.4332;
                 -7.7050, -7.7050, -7.5530];

%% Symbolicke promenne
nx = 12;
nu = 4;
X = SX.sym('X', nx, 1);
U = SX.sym('U', nu, 1);

u_stable = sqrt(g/(C_T*4));

%% Stavy
x = X(1); % pozice [x, y, z]
y = X(2);
z = X(3);
phi = X(4); % Eulerovy úhly [phi, theta, psi]
theta = X(5);
psi = X(6);
u = X(7); % rychlosti [vx, vy, vz]
v = X(8);
w = X(9);
p = X(10); % úhlové rychlosti [p, q, r]
q = X(11);
r = X(12);

%% Souradny system Zeme
E = eye(3);
e1 = E(:,1);
e2 = E(:,2);
e3 = E(:,3);

%% Rotacni matice
% Rotace kolem osy Z (yaw)
R_psi = [cos(psi), -sin(psi), 0;
         sin(psi),  cos(psi), 0;
         0,        0,        1];
     
% Rotace kolem osy Y (pitch)
R_theta = [cos(theta), 0, sin(theta);
           0,          1, 0;
          -sin(theta), 0, cos(theta)];
      
% Rotace kolem osy X (roll)
R_phi = [1, 0,         0;
         0, cos(phi), -sin(phi);
         0, sin(phi),  cos(phi)];
     
% Celková rotační matice
Rot = R_psi * R_theta * R_phi;
% Rot = [cos(theta), cos(theta)*sin(psi), -sin(theta);
%        sin(phi*sin(theta))*cos(psi)-cos(phi)*sin(psi), sin(phi*sin(theta))*sin(psi)+cos(phi)*cos(psi), sin(phi)*cos(theta);
%        cos(phi*sin(theta))*cos(psi)+sin(phi)*sin(psi), cos(phi*sin(theta))*sin(psi)-sin(phi)*cos(psi), cos(phi)*cos(theta)];

%% Translacni dynamika
Fz = C_T*sum(U.^2);
d_xyz = [u;v;w];
acc = -g*e3 + Rot*(Fz*e3)/m;    % [du;dv;dw]

%% Rotacni dynamika
J = diag([Ix, Iy, Iz]);
M = [(arm_length*C_T/sqrt(2))*(-U(1)^2-U(2)^2+U(3)^2+U(4)^2); ...
    (arm_length*C_T/sqrt(2))*(-U(1)^2+U(2)^2+U(3)^2-U(4)^2); ...
    C_M*(-U(1)^2+U(2)^2-U(3)^2+U(4)^2)];
ang_acc = inv(J)*(M - cross([p;q;r], J*[p;q;r]));

d_phi_theta_psi = [1, sin(phi)*tan(theta), cos(phi)*tan(theta);
                    0, cos(phi), -sin(phi);
                    0, sin(phi)/cos(theta), cos(phi)/cos(theta)]*[p;q;r];
%% Stavovy popis
dX = [d_xyz(1);
      d_xyz(2);
      d_xyz(3);
      acc(1);
      acc(2);
      acc(3);
      ang_acc(1);
      ang_acc(2);
      ang_acc(3);
      d_phi_theta_psi(1);
      d_phi_theta_psi(2);
      d_phi_theta_psi(3)
      ];

%% Omezeni
lbu = [rad_min; rad_min; rad_min; rad_min]-u_stable;  % input lower bounds
ubu = [rad_max; rad_max; rad_max; rad_max]-u_stable;  % input upper bounds

%% Vahove funkce - cost functions
val_q = 1;
val_u = 1;
% Q = (1/val_q^2)*eye(nx);    % state cost
Q = diag([1;1;1;0.3;0.3;0.6;1;1;1;0;0;0]);
% R = (1/val_u^2)*eye(nu);  % input cost
R = diag([0.3;0.3;0.3;0.3]);
% generic cost formulation
xr = [3;3;3;0;0;0;0;0;0;0;0;0]; % reference
% xr = [0:3/50:3;0:3/50:3;0:5/50:5;zeros(1,51);zeros(1,51);zeros(1,51);zeros(1,51);zeros(1,51);zeros(1,51);zeros(1,51);zeros(1,51);zeros(1,51)];
cost_expr_ext_cost_e = (X-xr)'*Q*(X-xr);  % terminal cost (only states)
cost_expr_ext_cost = cost_expr_ext_cost_e + U'*R*U;  % stage cost (states and inputs)
% cost_expr_ext_cost = 1/Ts * cost_expr_ext_cost;  % scale the stage cost to match the discrete formulation
% cost_expr_ext_cost_0 = 1/Ts * u'*R*u;  % penalize only the inputs in the first stage

%% Inicializace struktury modelu
model.X = X;
model.U = U;
model.dX = dX;
model.cost = cost_expr_ext_cost;
model.lbu = lbu;
model.ubu = ubu;
model.xr = xr;
model.u_stable = u_stable;
end

