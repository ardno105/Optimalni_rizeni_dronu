clc
clear
close all

%% Parametry
% kT = 3.16e-10; % rotor thrust coefficient
% ktau = 7.94e-12; % rotor torque coefficient
C_T = 1.28192e-8; % rotor thrust coefficient
C_M = 5.964552e-3; % rotor torque coefficient

rad_max = (2*pi*64*200)/60; % maximal angular rate of rotor [rad/s]
rad_min = (2*pi*64*18)/60; % minimal angular rate of rotor [rad/s]
rpm_max = rad_max*9.5492968;    % [rpm]
rpm_min = rad_min*9.5492968;    % [rpm]
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

%% Simulace
SimTime = 10;
