function Trajectory = Trajectory_generator(x0, xr, xt, vel_lim, acc_lim, N)
%TRAJECTORY_GENERATOR Summary of this function goes here
%   Detailed explanation goes here
% x0 - pocatecni bod [n x 1]
% xr - koncovy bod [n x 1]
% xt - body, kterymi chci proletet [n x m]
% vel_lim - omezeni rychlosti [lbx, ubx;
%                              lby, uby;
%                              lbz, ubz]
% acc_lim - omezeni zrycheni [lbx, ubx;
%                             lby, uby;
%                             lbz, ubz]
% N - pocet vzorku v trajektorii

waypoints = [x0, xt, xr];
vellim = vel_lim;
accellim = acc_lim;
[q,qd,qdd,t] = contopptraj(waypoints,vellim,accellim, "NumSamples",N);
Trajectory = q;
end

