
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



# Heat Conduction Simulation
function heat_conduction!(dθ, θ, param, t)
    u_in = param[1:3]
    diffusion!(dθ, θ, plate, property, boundary, actuation, u_in)
end

Tf = 60 # 1200;
tspan = (0.0, Tf)
θ₀ = 500;
Ntotal = N₁*N₂
θinit = θ₀*ones(Ntotal)

dΘ = similar(θinit)
heat_conduction!(dΘ,θinit,ones(3),0)


# Δt = 1e-2               # Sampling time

using OrdinaryDiffEq

p_input = [8e3,4e3,8e3];

t_samp = 10.0
alg = KenCarp5()
prob_orig = ODEProblem(heat_conduction!,θinit,tspan)
sol = solve(prob_orig,alg,p=p_input, saveat=t_samp)
θsol = Array(sol)



using Plots
plot(sol,legend=false)
plot(θsol[N₁*(N₂-1)+1:end,:]',legend=false)


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
    pars = u
    sol_loss = solve(prob, alg,p=pars, saveat = t_samp)
    if sol_loss.retcode == ReturnCode.Success
        y = C*Array(sol_loss)
        err = ref_data .- y
        loss = sum(abs2, err) *t_samp / Tf
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

t_mpc = 30; # MPC sampling time
N_mpc = 3; # MPC control horizon















loss_optim(p_input, [0],prob_orig)

p_data_store = zeros(3,10);

using Optimization, OptimizationOptimJL, ForwardDiff

# prob_ode = remake(prob)

loss_fun(u,p)  = loss_optim(u,p,prob_ode)
loss_fun(p_input, [0])

optf = OptimizationFunction(loss_fun, Optimization.AutoForwardDiff())

opt_prob = OptimizationProblem(optf, p_input, [0])
p_opt = Optimization.solve(opt_prob, ConjugateGradient(), callback=callback, maxiters=3)
sol_temp = solve(prob_ode,alg,p=p_opt, saveat=t_samp)


prob_ode = remake(prob_ode; tspan = tspan .+ 60, u0 = sol_temp[:,end])
loss_optim(p_input,[0], prob_ode)
p_data_store[:,1] = p_opt


sol_temp = solve(prob_ode,alg,p=zeros(3), saveat=t_samp)

plot(sol_temp, legend=false)
temp22 = (C*Array(sol_temp))'
plot(temp22,legend=false)


solve(prob_ode,alg,p=zeros(3), save_everystep=false)














sol_12 = solve(prob,alg,p=p_opt, saveat=t_samp)

θsol12 = Array(sol_12)
plot(θsol12[N₁*(N₂-1)+1:end,:]',legend=false)

C*θsol12


n_ts = length(sol_12.t)
θsol_2d = reshape(θsol12, N₁,N₂,n_ts)
contourf(θsol_2d[:,N₂,:])

p_in2 = 3e3*ones(3); # [5845, 3897, 5845]
prob1 = remake(prob; tspan = tspan .+ 60, u0 = θsol12[:,end])
sol_22 = solve(prob1,alg,p=p_in2, saveat=t_samp)

temp22 = (C*Array(sol_22))'
plot(temp22,legend=false)

