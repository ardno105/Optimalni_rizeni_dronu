clc
clear
close all
%% Parametry dronu Crazyflie
addpath('C:\Users\Ondra\VS\bakalarka\CasADi');
import casadi.*

random_matrix = DM.eye(3).*[4;3;2]
Ts = 1e-3;  % perioda vzorkovani

N = 100;    % pocet intervalu
Xr = [1;1;1;0;0;0;0;0;0;0;0;0]; % reference

%% Model dronu
model = drone_model();
% model = drone_model2();

%% Inicializace stavu, vstupu, rovnic a omezeni
% stavy 12x1
X = model.X;

% vstupy 4x1
U = model.U;

% rovnice dynamiky
f = model.dX;

% omezeni
h = model.cost;

dae = struct('x', X, 'p', U, 'ode', f, 'quad', h);

%% Vytvoreni solveru
T = 10; % koneecny cas
N = 20; % diskretizace

% Nastaveni integratoru (nazev, typ, rovnice, odkud, kam)
op = struct('t0',0,'tf',T/N);
F = integrator('F','idas',dae,0,T/N);

% Prazdny LNP
w={}; lbw=[]; ubw=[];
G={}; J=0;
% Pocatecni podminky
Xk = MX.sym('X0',12);
w{end+1} = Xk;
x0 = zeros(12,1);

% omezeni prostoru
ubx = inf*ones(12,1);   
lbx = -inf*ones(12,1);
% pocatecni nastaveni
lbw = [lbw;x0];
ubw = [ubw;x0];

for k=1:N
    % Local control
    Uname = ['U' num2str(k-1)];
    Uk = MX.sym(Uname,4,1);
    w{end+1} = Uk;
    % nastaveni omezeni vstupu
    lbw = [lbw; model.lbu];
    ubw = [ubw; model.ubu];
    % integrator
    Fk = F('x0',Xk,'p',Uk);
    J = J+Fk.qf;
    % novy stav
    Xname = ['X' num2str(k)];
    Xk = MX.sym(Xname,12,1);
    w{end+1} = Xk;
    lbw = [lbw;lbx];
    ubw = [ubw; ubx];
    % Continuity constraint
    G{end+1} = Fk.xf-Xk;
end

%% NLP solver
nlp = struct('f',J,'g',vertcat(G{:}),'x',vertcat(w{:}));
S = nlpsol('S','ipopt',nlp);
%% Reseni NLP
r = S('lbx',lbw,'ubx', ubw,'x0',0,'lbg',0,'ubg',0);
disp(r.x);

%% Vykresleni
figure;
hold on
plot(full(r.x(3:16:end)))   % vykresleni vyvoje v ose z
plot(full(r.x(1:16:end)))
plot(full(r.x(2:16:end)))
