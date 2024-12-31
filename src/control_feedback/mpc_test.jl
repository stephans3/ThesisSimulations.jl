

function input(p, t)
    if t <= 0
        return zeros(size(p)[1])
    else
        n = ceil(Int64, t/t_mpc)
        return p[:,n]
    end
end


function ode!(dx,x,p,t)
    u = input(p,t)

    dx[1] = -x[1] + u[1]
    dx[2] = -x[2] + u[2]
    dx[3] = -x[3] + u[3]
end
    

using OrdinaryDiffEq
t_mpc = 3;
N_mpc = 3;
Tf = t_mpc*N_mpc;  #60 # 1200;
tspan = (0.0, Tf)

pinit = repeat([1.0,2.0,3.0],1,3);

x0 = 10*ones(3)
t_samp = 10.0
alg = KenCarp5()
prob_ode = ODEProblem(ode!,x0,tspan)
sol = solve(prob_ode,alg,p=pinit, saveat=t_mpc)


function loss_optim(u,p,prob)
    prob = prob
    pars = u
    sol_loss = solve(prob, alg,p=pars, saveat = t_mpc)
    if sol_loss.retcode == ReturnCode.Success
        err = 2.5 .- Array(sol_loss)
        loss = sum(abs2, err)
       
    else
        loss = Inf
    end
    return loss
end

loss_optim(pinit,[],prob_ode)

using Optimization, OptimizationOptimJL, ForwardDiff

loss_fun(u,p)  = loss_optim(u,p,prob_ode)
loss_fun(pinit, [])

optf = OptimizationFunction(loss_fun, Optimization.AutoForwardDiff())

lower_bounds = 0.1ones(3,N_mpc) #zeros(3,N_mpc)
upper_bounds = 1e10*ones(3,N_mpc)

opt_prob = OptimizationProblem(optf, pinit, [0], lb=lower_bounds, ub=upper_bounds)
p_opt = Optimization.solve(opt_prob, BFGS(), maxiters=3)
sol_temp = solve(prob_ode,alg,p=Array(p_opt), saveat=t_samp)
