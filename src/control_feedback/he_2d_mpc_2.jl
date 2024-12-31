
λ₁ = 40;
λ₂ = 60;
ρ = 8000; # Density
c = 400;  # Specific heat capacity

L = 0.3; # Length
W = 0.05
N₁ = 30;
N₂ = 10;
Nc = N₁*N₂ 
Δx₁ = L/N₁
Δx₂ = W/N₂


using Hestia 
property = StaticAnisotropic([λ₁,λ₂], ρ,c)
plate  = HeatPlate(L,W, N₁,N₂)
boundary = Boundary(plate)

### Boundaries ###
θamb = 300.0;
h = 10;
ε = 0.1;
emission = Emission(h, θamb,ε) #  Stefan-Boltzmann BC: heat transfer (linear) and heat radiation (quartic/nonlinear)
boundary = Boundary(plate)

setEmission!(boundary, emission, :west )
setEmission!(boundary, emission, :east )
setEmission!(boundary, emission,  :north )

### Actuation ###
actuation = IOSetup(plate)
num_actuators = 3        # Number of actuators per boundary
config  = RadialCharacteristics(1.0, 2, 30)
setIOSetup!(actuation, plate, num_actuators, config, :south)

actuator_char = getCharacteristics(actuation, :south)[1]


# t_mpc = 30;
# N_mpc = 3;
# -> t_sim = 3*30 = 90
# t/t_sim


function input(p, t)
    if t <= 0
        return zeros(size(p)[1])
    else
        n = ceil(Int64, t/t_mpc)
        nstart = 3*(n-1) + 1
        nend = nstart + 2
        return p[nstart:nend]
    end
end

# p_in = [1 3 5; 2 4 6; 7 8 9]
# tgrid = 0:0.1:N_mpc*t_mpc
# map(t-> input(p_in,t),tgrid)

# Heat Conduction Simulation
function heat_conduction!(dθ, θ, param, t)
    u_in = input(param, t)
    diffusion!(dθ, θ, plate, property, boundary, actuation, u_in)
end


t_mpc = 30;
N_mpc = 3;
Tf = t_mpc*N_mpc;  #60 # 1200;
tspan = (0.0, Tf)
θ₀ = 500;
Ntotal = N₁*N₂
θinit = θ₀*ones(Ntotal)

# dΘ = similar(θinit)
# heat_conduction!(dΘ,θinit,ones(3),0)


# Δt = 1e-2               # Sampling time

using OrdinaryDiffEq

pinit = vcat([15e3,12e3,15e3], [3e3,1e3,3e3],[6e3,3e3,6e3])

t_samp = 10.0
alg = KenCarp5()
prob_orig = ODEProblem(heat_conduction!,θinit,tspan)
sol = solve(prob_orig,alg,p=pinit, saveat=t_mpc)

#=
θsol = Array(sol)

using Plots
plot(sol,legend=false)
plot(θsol[N₁*(N₂-1)+1:end,:]',legend=false)
=#

### Sensor ###
num_sensor = 3        # Number of sensors
sensing = IOSetup(plate)
config_sensor  = RadialCharacteristics(1.0, 1, 0)
setIOSetup!(sensing, plate, num_sensor, config_sensor, :north)

sensor_char = getCharacteristics(sensing, :north)[1]
C = hcat(zeros(3,N₁*(N₂-1)), sensor_char' ./  sum(sensor_char, dims=1)')
ref_data = 500*ones(3);


function loss_optim(u,p,prob)
    prob = prob
    pars = exp.(u)
    N_u = num_actuators;
    sol_loss = solve(prob, alg,p=pars, saveat = t_mpc)
    if sol_loss.retcode == ReturnCode.Success
        y = C*Array(sol_loss)
        err = ref_data .- y
        loss_y = sum(abs2, err)
        pars_arr = reshape(pars, N_u, N_mpc)
        loss_u = (1e-7)*sum(abs2, pars_arr[:,1:end-1] - pars_arr[:,2:end])/((N_mpc-1)*N_u)
        loss = loss_y + loss_u
    else
        loss = Inf
    end
    return loss
end

const store_loss=[]
global store_param=[]

callback = function (state, l) 
    # store loss and parameters
    append!(store_loss,l) # Loss values 
    
    # store_param must be global
    global store_param  
    store_param = vcat(store_param,[state.u]) # save with vcat


    println(l)
    #println("iter")

    if l < -0.2
        return true
    end

    return false
end

# t_mpc = 30; # MPC sampling time
# N_mpc = 3; # MPC control horizon

N_mpc_samples = 10
p_data_store = zeros(3,N_mpc_samples);
pinit = 8*ones(3*N_mpc)

loss_optim(pinit,[],prob_orig)

using Optimization, OptimizationOptimJL, ForwardDiff

prob_ode = remake(prob_orig)
loss_fun(u,p)  = loss_optim(u,p,prob_ode)
optf = OptimizationFunction(loss_fun, Optimization.AutoForwardDiff())
opt_prob = OptimizationProblem(optf, pinit, [0])
for n=1:N_mpc_samples
    p_opt = Optimization.solve(opt_prob, ConjugateGradient(), maxiters=3)
    sol_temp = solve(prob_ode,alg,p=exp.(Array(p_opt)), save_everystep=false)
    prob_ode = remake(prob_ode; u0 = sol_temp[:,end])
    p_opt_data = Array(p_opt)
    p_data_store[:,n] = p_opt_data[1:3]
    p_new = vcat(p_opt_data[4:9],p_opt_data[7:9])

    loss_fun(u,p) = loss_optim(u,p,prob_ode)
    optf = OptimizationFunction(loss_fun, Optimization.AutoForwardDiff())
    opt_prob = remake(opt_prob; f=optf, u0=p_new)
end

#=
p_opt_data = [
9.51674  8.69653  8.88635  8.63213  8.83793  8.74208  8.79865  8.75205  8.79137  8.77228
 9.18024  7.97067  7.57852  7.59994  7.82952  7.81358  7.83333  7.82207  7.84787  7.84244
 9.51674  8.69653  8.88635  8.63213  8.83793  8.74208  8.79865  8.75205  8.79137  8.77228]
=#

exp.(p_data_store)


p_store_vec = exp.(reshape(p_data_store,3*N_mpc_samples))





Tf = t_mpc*N_mpc_samples;  #60 # 1200;
tspan = (0.0, Tf)
θ₀ = 500;
θinit = θ₀*ones(Ntotal)

t_samp = 1.0
alg = KenCarp5()
prob_demo = ODEProblem(heat_conduction!,θinit,tspan)
sol_demo = solve(prob_demo,alg,p=p_store_vec, saveat=t_samp)

using Plots
plot(sol_demo)


temp22 = (C*Array(sol_demo))'
plot(temp22,legend=false)










prob_ode = remake(prob_orig)
loss_fun(u,p)  = loss_optim(u,p,prob_ode)
loss_fun(pinit, [])

optf = OptimizationFunction(loss_fun, Optimization.AutoForwardDiff())

#lower_bounds = zeros(3*N_mpc) #-Inf*ones(3,Nmpc)
#upper_bounds = Inf*ones(3*N_mpc)#20ones(3,Nmpc)

opt_prob = OptimizationProblem(optf, pinit, [0])#, lb=lower_bounds, ub=upper_bounds)
p_opt = Optimization.solve(opt_prob, ConjugateGradient(), callback=callback, maxiters=3)
sol_temp = solve(prob_ode,alg,p=exp.(Array(p_opt)), saveat=t_samp)

prob_ode = remake(prob_ode; u0 = sol_temp[:,end])
p_data_store[:,1] = Array(p_opt)[1:3,1]

loss_fun(u,p)  = loss_optim(u,p,prob_ode)
optf = OptimizationFunction(loss_fun, Optimization.AutoForwardDiff())

p_opt_data = Array(p_opt)
p_init_new = vcat(p_opt_data[4:9],p_opt_data[7:9])
opt_prob = remake(opt_prob; f=optf, u0=p_init_new)

#loss_optim(p_init_new,[],prob_ode)

p_opt2 = Optimization.solve(opt_prob, ConjugateGradient(), callback=callback, maxiters=3)
sol_temp = solve(prob_ode,alg,p=p=exp.(Array(p_opt2)), saveat=t_samp)

exp.(Array(p_opt2))


p_opt_data2 = Array(p_opt2)
(1e-7)*sum(abs2, p_opt_data[:,1:end-1] - p_opt_data[:,2:end])/((N_mpc-1)*3)
(1e-7)*sum(abs2, p_opt_data2[:,1:end-1] - p_opt_data2[:,2:end])/((N_mpc-1)*3)


(1e-2)*(p_opt_data2[:,1] - p_opt_data2[:,2])
(1e-3)*(p_opt_data2[:,2] - p_opt_data2[:,3])


using Plots
plot(sol_temp,legend=false)
temp22 = (C*Array(sol_temp))'
plot(temp22,legend=false)