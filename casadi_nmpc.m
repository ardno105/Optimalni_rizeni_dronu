classdef casadi_nmpc < matlab.System & matlab.system.mixin.Propagates
    % untitled Add summary here
    %
    % This template includes the minimum set of functions required
    % to define a System object with discrete state.

    properties
        % Public, tunable properties.

    end

    properties (DiscreteState)
    end

    properties (Access = private)
        % Pre-computed constants.
        casadi_solver
        % x0
        % lbx
        % ubx
        % lbg
        % ubg
        % F_koef
        model
    end

    methods (Access = protected)
        function num = getNumInputsImpl(~)
            num = 3;
        end
        function num = getNumOutputsImpl(~)
            num = 1;
        end
        function dt1 = getOutputDataTypeImpl(~)
        	dt1 = 'double';
        end
        function dt1 = getInputDataTypeImpl(~)
        	dt1 = 'double';
        end
        function sz1 = getOutputSizeImpl(~)
        	sz1 = [4,1];
        end
        function sz1 = getInputSizeImpl(~)
        	sz1 = [1,1];
        end
        function cp1 = isInputComplexImpl(~)
        	cp1 = false;
        end
        function cp1 = isOutputComplexImpl(~)
        	cp1 = false;
        end
        function fz1 = isInputFixedSizeImpl(~)
        	fz1 = true;
        end
        function fz1 = isOutputFixedSizeImpl(~)
        	fz1 = true;
        end
        function setupImpl(obj,~,~)
            % Implement tasks that need to be performed only once, 
            % such as pre-computed constants.
            
            addpath('C:\Users\Ondra\VS\bakalarka\CasADi');
            import casadi.*
            
            % Model dronu
            % Parametry
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
            % Symbolicke promenne
            nx = 12;
            nu = 4;
            X = SX.sym('X', nx, 1);
            U = SX.sym('U', nu, 1);
            xr = SX.sym('xr', nx, 1);
            
            u_stable = sqrt(g/(C_T*4));

            % Stavy
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
            
            % Souradny system Zeme
            E = eye(3);
            e1 = E(:,1);
            e2 = E(:,2);
            e3 = E(:,3);
            
            % Rotacni matice
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
            
            % Translacni dynamika
            Fz = C_T*sum(U.^2);
            d_xyz = [u;v;w];
            acc = -g*e3 + Rot*(Fz*e3)/m;    % [du;dv;dw]
            
            % Rotacni dynamika
            J = diag([Ix, Iy, Iz]);
            M = [(arm_length*C_T/sqrt(2))*(-U(1)^2-U(2)^2+U(3)^2+U(4)^2); ...
                (arm_length*C_T/sqrt(2))*(-U(1)^2+U(2)^2+U(3)^2-U(4)^2); ...
                C_M*(-U(1)^2+U(2)^2-U(3)^2+U(4)^2)];
            ang_acc = inv(J)*(M - cross([p;q;r], J*[p;q;r]));
            
            d_phi_theta_psi = [1, sin(phi)*tan(theta), cos(phi)*tan(theta);
                                0, cos(phi), -sin(phi);
                                0, sin(phi)/cos(theta), cos(phi)/cos(theta)]*[p;q;r];
            % Stavovy popis
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
            
            % Omezeni
            lbu = [rad_min; rad_min; rad_min; rad_min]-u_stable;  % input lower bounds
            ubu = [rad_max; rad_max; rad_max; rad_max]-u_stable;  % input upper bounds
            
            model_init = struct();
            model_init.X = X;
            model_init.U = U;
            model_init.dX = dX;
            model_init.lbu = lbu;
            model_init.ubu = ubu;
            model_init.u_stable = u_stable;
            model_init.F_koef = C_T;

            obj.model = model_init;



            % obj.casadi_solver = solver;
            % obj.x0 = w0;
            % obj.lbx = lbw;
            % obj.ubx = ubw;
            % obj.lbg = lbg;
            % obj.ubg = ubg;
            % obj.F_koef = C_T;
        end

        function control = stepImpl(obj,state,ref,t)
            import casadi.*
            disp(t)
            tic
            % w0 = obj.x0;
            % lbw = obj.lbx;
            % ubw = obj.ubx;
            % C_T = obj.F_koef;
            % solver = obj.casadi_solver;
            model = obj.model;
            xr = [ref; zeros(9,1)];
            C_T = model.F_koef;
            % lbw(1:12) = state;
            % ubw(1:12) = state;

            T = 2; % Casovy horizont
            N = 6; % Pocet intervalu
            % Pomer T/N = 1/3 vypada celkem dobre
            
            % Promenne
            x = model.X;    % Stav
            u = model.U;    % Vstup
            
            % Rovnice modelu
            xdot = model.dX;
            
            % Objectivni funkce
            Q = diag([1;1;1;0.3;0.3;0.6;1;1;1;0;0;0]);
            R = diag([0.3;0.3;0.3;0.3]);
            L = SX.sym('L',N,1);
            for k = 1:N
                L(k) = (x-xr)'*Q*(x-xr) + u'*R*u;  % stage cost (states and inputs)
            end
            % L = model.cost;
            
            % Continuous time dynamics
            % f = Function('f', {x, u}, {xdot, L});
            
            % Formulate discrete time dynamics
            F = {};
            if true
               for k=1:N
                    % CVODES from the SUNDIALS suite
                   dae = struct('x',x,'p',u,'ode',xdot,'quad',L(k));
                   opts = struct('tf',T/N);
                   F{end+1} = integrator('F', 'cvodes', dae, 0, 1/(T/N));
               end
               
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
            lbw = [lbw; state];
            ubw = [ubw; state];
            % lbw = [lbw; [3;3;3;0;0;0;0;0;0;0;0;0]];
            % ubw = [ubw; [3;3;3;0;0;0;0;0;0;0;0;0]];
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
                fcn = F{k+1};
                Fk = fcn('x0', Xk, 'p', Uk);
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
                lbg = [lbg; [zeros(3,1);-inf(9,1)]];
                ubg = [ubg; [zeros(3,1);inf(9,1)]];
            end
            
            % NLP nastaveni
            opts = struct;
            opts.ipopt.max_iter = 50;
            opts.ipopt.acceptable_tol =1e-3;
            opts.ipopt.acceptable_obj_change_tol = 1e-3;
            
            % NLP solver
            prob = struct('f', J, 'x', vertcat(w{:}), 'g', vertcat(g{:}));
            solver = nlpsol('solver', 'ipopt', prob, opts);
            
            % Vyres NLP
            sol = solver('x0', w0, 'lbx', lbw, 'ubx', ubw,...
                        'lbg', lbg, 'ubg', ubg);
            phi = full(sol.x(7));
            theta = full(sol.x(8));
            psi = full(sol.x(12));
            u_input = [full(sol.x(13));
                       full(sol.x(14));
                       full(sol.x(15));
                       full(sol.x(16))];
            Fz = 3.826e-6*sum(u_input.^2);
            control = [phi; theta; psi; Fz];
            toc
        end

        function resetImpl(obj)
            % Initialize discrete-state properties.
        end
    end
end
