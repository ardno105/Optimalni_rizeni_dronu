clc
clear
close all
%% Parametry dronu Crazyflie
addpath('C:\Users\Ondra\VS\bakalarka\CasADi');
import casadi.*

model = drone_model();

T = 0.5; % Casovy horizont
N = 100; % Pocet intervalu
% Pomer T/N = 1/3 vypada celkem dobre

% Promenne
x = model.X;    % Stav
u = model.U;    % Vstup
xr_sym = SX.sym('xr', 12,1);


%% Generator trajektorie
% Konecny bod
XR = [1;1;1];
x0 = zeros(3,1);

% Omezeni rychlosti a zrychleni
vel_lim = [-100*ones(3,1), 100*ones(3,1)];
acc_lim = [-100*ones(3,1), 100*ones(3,1)];

% Body, kterymi chci proletet
xt = [];
xr = Trajectory_generator(x0, XR, xt, vel_lim, acc_lim, N);
xr = [XR;
      0;
      0;
      0;
      0;
      0;
      0;
      0;
      0;
      0;
      ];
% xr = [xr;
%       ones(1,N-1), 0;
%       ones(1,N-1), 0;
%       ones(1,N-1), 0;
%       zeros(1,N);
%       zeros(1,N);
%       zeros(1,N);
%       zeros(1,N);
%       zeros(1,N);
%       zeros(1,N);
%       ];

% Rovnice modelu
xdot = model.dX;

% Objectivni funkce
Q = diag([1;1;1;0.3;0.3;0.6;1;1;1;0;0;0]);
% Q = eye(12);
R = diag([0.3;0.3;0.3;0.3]);
% R = diag([1;1;1;1]);
% L = SX.sym('L',N,1);
% for k = 1:N
%     L(k) = (x-xr(:,k))'*Q*(x-xr(:,k)) + u'*R*u;  % stage cost (states and inputs)
% end
L = (x-xr_sym)'*Q*(x-xr_sym) + u'*R*u;
% L = model.cost;

% Continuous time dynamics
% f = Function('f', {x, u}, {xdot, L});

% Formulate discrete time dynamics
% F = {};
dae = struct('x',x, 'p',[u;xr_sym],'ode',xdot,'quad',L);
opts = struct('tf',T/N, 'allow_free', true);
% F = integrator('F', 'cvodes', dae, 0, 1/(T/N));
F = integrator('F', 'cvodes', dae, 0, T);
% if true
%    for k=1:N
%         % CVODES from the SUNDIALS suite
       % dae = struct('x',x,'p',u,'ode',xdot,'quad',L);
       % opts = struct('tf',T/N);
%        F{end+1} = integrator('F', 'cvodes', dae, 0, 1/(T/N));
%    end
% 
% else
%    % Fixed step Runge-Kutta 4 integrator
%    M = 4; % RK4 steps per interval
%    DT = T/N/M;
%    f = Function('f', {x, u}, {xdot, L});
%    X0 = MX.sym('X0', 12);
%    U = MX.sym('U',4);
%    X = X0;
%    Q = 0;
%    for j=1:M
%        [k1, k1_q] = f(X, U);
%        [k2, k2_q] = f(X + DT/2 * k1, U);
%        [k3, k3_q] = f(X + DT/2 * k2, U);
%        [k4, k4_q] = f(X + DT * k3, U);
%        X=X+DT/6*(k1 +2*k2 +2*k3 +k4);
%        Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q);
%     end
%     F = Function('F', {X0, U}, {X, Q}, {'x0','p'}, {'xf', 'qf'});
% end

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
lbw = [lbw; zeros(12,1)];
ubw = [ubw; zeros(12,1)];
% lbw = [lbw; [3;3;3;0;0;0;0;0;0;0;0;0]];
% ubw = [ubw; [3;3;3;0;0;0;0;0;0;0;0;0]];
w0 = [w0; zeros(12,1)];

% Formulace NLP
for k=0:N-1
    % Vstupy
    Uk = MX.sym(['U_' num2str(k)],4);
    w = {w{:}, Uk};
    % Omezeni vstupu
    % lbw = [lbw; model.u_stable + model.lbu];
    % ubw = [ubw; model.u_stable + model.ubu];
    lbw = [lbw; model.lbu];
    ubw = [ubw; model.ubu];
    w0 = [w0; zeros(4,1)];
    % w0 = [w0; ones(4,1)*model.u_stable];

    % Integrace
    % fcn = F{k+1};
    fcn = F;
    % Fk = fcn('x0', Xk, 'p', [Uk;xr(:,k+1)]);
    Fk = fcn('x0', Xk, 'p', [Uk;xr]);
    Xk_end = Fk.xf;
    J=J+Fk.qf;
    % J = substitute(J, xr_sym, xr(:,k+1));

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
    lbg = [lbg; [zeros(3,1);-inf(9,1)]];
    ubg = [ubg; [zeros(3,1);inf(9,1)]];
end

% NLP nastaveni
opts = struct;
opts.ipopt.max_iter = 10;
opts.ipopt.acceptable_tol =1e-3;
opts.ipopt.acceptable_obj_change_tol = 1e-3;

% NLP solver
% prob = struct('f', J, 'x', vertcat(w{:}), 'g', vertcat(g{:}));
prob = struct('f', J, 'x', vertcat(w{:}), 'g', vertcat(g{:}));
solver = nlpsol('solver', 'ipopt', prob, opts);

% Vyres NLP
% lbw_short = [lbw(1:16); ]
sol = solver('x0', w0, 'lbx', lbw, 'ubx', ubw,...
            'lbg', lbg, 'ubg', ubg);
w_opt = full(sol.x);

%% Vykresleni
% xxr = [xr(1,:), xr(1,end)];
% yxr = [xr(2,:), xr(2,end)];
% zxr = [xr(3,:), xr(3,end)];
xxr = [ones(1,N+1)*xr(1)];
yxr = [ones(1,N+1)*xr(2)];
zxr = [ones(1,N+1)*xr(3)];

% Poloha 3D
figure
hold on
plot3(w_opt(1:16:end),w_opt(2:16:end),w_opt(3:16:end))
plot3(xxr,yxr,zxr, 'r--')
xlabel('x');
ylabel('y');
zlabel('z')
xlim([0 3])
ylim([0 3])
zlim([0 3])

% Poloha 2D
figure
title('Poloha')
subplot(3,1,1)
    hold on
    plot(w_opt(1:16:end))
    plot(xxr, 'r--')
    legend('x', 'x_t')
    xlabel('N')
    ylabel('x')
    xlim([1 N+1])
subplot(3,1,2)
    hold on
    plot(w_opt(2:16:end))
    plot(yxr, 'r--')
    legend('y', 'y_t')
    xlabel('N')
    ylabel('y')
    xlim([1 N+1])
subplot(3,1,3)
    hold on
    plot(w_opt(3:16:end))
    plot(zxr, 'r--')
    legend('z', 'z_t')
    xlabel('N')
    ylabel('z')
    xlim([1 N+1])


% Vstupy
figure
hold on
plot(w_opt(13:16:end))
plot(w_opt(14:16:end))
plot(w_opt(15:16:end))
plot(w_opt(16:16:end))
title('Vstupy')
legend('u1', 'u2', 'u3', 'u4')
xlim([1 N+1])
