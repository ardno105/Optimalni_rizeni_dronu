clc
clear
close all
%% Parametry dronu Crazyflie
addpath('C:\Users\Ondra\VS\bakalarka\CasADi');
import casadi.*

model = drone_model2();

T = 10; % Casovy horizont
N = 50; % Pocet intervalu

% Promenne
x = model.X;    % Stav
u = model.U;    % Vstup

% Rovnice modelu
xdot = model.dX;

% Objectivni funkce
L = model.cost;

% Continuous time dynamics
f = Function('f', {x, u}, {xdot, L});

% Formulate discrete time dynamics
if true
   % CVODES from the SUNDIALS suite
   dae = struct('x',x,'p',u,'ode',xdot,'quad',L);
   opts = struct('tf',T/N);
   F = integrator('F', 'cvodes', dae, 0, T/N);
else
   % Fixed step Runge-Kutta 4 integrator
   M = 4; % RK4 steps per interval
   DT = T/N/M;
   f = Function('f', {x, u}, {xdot, L});
   X0 = MX.sym('X0', 12);
   U = MX.sym('U',4);
   X = X0;
   Q = 0;
   for j=1:M
       [k1, k1_q] = f(X, U);
       [k2, k2_q] = f(X + DT/2 * k1, U);
       [k3, k3_q] = f(X + DT/2 * k2, U);
       [k4, k4_q] = f(X + DT * k3, U);
       X=X+DT/6*(k1 +2*k2 +2*k3 +k4);
       Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q);
    end
    F = Function('F', {X0, U}, {X, Q}, {'x0','p'}, {'xf', 'qf'});
end

% Inicializace NLP
w={};
w0 = [];
lbw = [];
ubw = [];
J = 0;
g={};
lbg = [];
ubg = [];

% Zavedeni pocatecnich podminek
Xk = MX.sym('X0', 12);
w = {w{:}, Xk};
% lbw = [lbw; 0; 1];
% ubw = [ubw; 0; 1];
% w0 = [w0; 0; 1];
lbw = [lbw; zeros(12,1)];
ubw = [ubw; zeros(12,1)];
w0 = [w0; zeros(12,1)];

% Formulace NLP
for k=0:N-1
    % Vstupy
    Uk = MX.sym(['U_' num2str(k)],4);
    w = {w{:}, Uk};
    % Omezeni vstupu
    lbw = [lbw; model.u_stable + model.lbu];
    ubw = [ubw; model.u_stable + model.ubu];
    w0 = [w0; zeros(4,1)];

    % Integrace
    Fk = F('x0', Xk, 'p', Uk);
    Xk_end = Fk.xf;
    J=J+Fk.qf;

    % Novy stav na konci integrace
    Xk = MX.sym(['X_' num2str(k+1)], 12);
    w = [w, {Xk}];
    % Omezeni stavu
    lbw = [lbw; -1*inf(12,1)];
    ubw = [ubw;  inf(12,1)];
    w0 = [w0; zeros(12,1)];

    % Rovnost stavu na konci integrace
    % g = {g{:}, Xk_end-Xk};
    g = [g, {Xk_end - Xk}];
    lbg = [lbg; zeros(12,1)];
    ubg = [ubg; 5*ones(12,1)];
end

% NLP solver
prob = struct('f', J, 'x', vertcat(w{:}), 'g', vertcat(g{:}));
solver = nlpsol('solver', 'ipopt', prob);

% Vyres NLP
sol = solver('x0', w0, 'lbx', lbw, 'ubx', ubw,...
            'lbg', lbg, 'ubg', ubg);
w_opt = full(sol.x);

%% Vykresleni
% Poloha
figure
hold on
plot(w_opt(1:16:end))
plot(w_opt(2:16:end))
plot(w_opt(3:16:end))
title('Poloha')
legend('x', 'y', 'z')
xlim([0 N])


% Vstupy
figure
hold on
plot(w_opt(13:16:end))
plot(w_opt(14:16:end))
plot(w_opt(15:16:end))
plot(w_opt(16:16:end))
title('Vstupy')
legend('u1', 'u2', 'u3', 'u4')
xlim([0 N])
