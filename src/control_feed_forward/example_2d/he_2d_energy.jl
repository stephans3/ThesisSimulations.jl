


# Compute thermal conductivity
# x₁ direction

θvec = collect(300:50:500)
M_temp = mapreduce(z-> [1  z  z^2 z^3 z^4], vcat, θvec)

λ1data = [40,44,50,52,52.5]
λ1p = inv(M_temp)*λ1data

λ2data = [40,55,60,65,68]
λ2p = inv(M_temp)*λ2data

ρ = 8000; # Density
c = 400;  # Specific heat capacity

L = 0.3; # Length of 1D rod
W = 0.05
N₁ = 12;
N₂ = 10;
Nc = N₁*N₂ 
Δx₁ = L/N₁
Δx₂ = W/N₂

using Hestia 

# property = StaticAnisotropic(40, 40, ρ,c)

property = DynamicAnisotropic(λ1p, λ2p, [ρ],[c])

plate  = HeatPlate(L,W, N₁,N₂)
boundary = Boundary(plate)

### Boundaries ###
θamb = 300.0;
h = 10;
Θamb = 300;
ε = 0.1;
emission = Emission(h, Θamb,ε) #  Stefan-Boltzmann BC: heat transfer (linear) and heat radiation (quartic/nonlinear)
boundary = Boundary(plate)

setEmission!(boundary, emission, :west )
setEmission!(boundary, emission, :east )
setEmission!(boundary, emission,  :north )

### Actuation ###
actuation = IOSetup(plate)
num_actuators = 3        # Number of actuators per boundary
config  = RadialCharacteristics(1.0, 1, 0)
setIOSetup!(actuation, plate, num_actuators, config, :south)

function input_obc(t,p)
    return exp(p[1] - p[3]^2 * (t/Tf - 1/p[2])^2)
end

Tf = 1200;
ts = (Tf/1000)
tgrid = 0 : ts : Tf

p_fbc = [11.81522609265776
            2.070393374741201
            9.212088025161702]

# Reference
ref_init = 300;
Δr = 200; # Difference operating points
ps = 10; # Steepness
ψ(t) = (1+tanh(10*(t/Tf - 1/2)))/2
ref(t) = ref_init + Δr*ψ(t)

# Input Signal
#=
input_u(t) = input_obc(t,p_fbc)
input_data = input_u.(tgrid)
input_int = sum(input_data[2:end])*ts
E_in = 0.3*input_int
=#


ΔU = ρ *c *L*W*Δr

E_tr = (h*(ts*sum(ref.(tgrid[2:end])) - Tf*Θamb))
coeff_rad = ε*5.67*1e-8;
E_rad = (coeff_rad*ts*sum(ref.(tgrid[2:end]).^4))
E_em_approx =  (L+2W)*(E_tr + E_rad)

using SpecialFunctions
u_in_energy(p₁,p₂,p₃) = exp(p₁)*Tf*sqrt(pi) * (erf(p₃-p₃/p₂) - erf(-p₃/p₂)) / (2 * p₃)
u_in_energy(p_fbc...)

act_sc_inv = 1/0.3 # Inverse of spatial characteristics of actuator
E_oc_mean = act_sc_inv * ΔU

u₀ = 1e-1;
p₂ = p_fbc[2]
function energy_em_approx!(F, x)
    F[1] = u₀- input_obc(0,[x[1], p₂, x[2]])
    F[2] = E_em_approx + E_oc_mean - u_in_energy(x[1], p₂, x[2])
end

using NLsolve
sol_nl_em_approx = nlsolve(energy_em_approx!, [p_fbc[1],p_fbc[2]])
p12,p32 = sol_nl_em_approx.zero

p_energy = [p12,p₂,p32] 


function heat_conduction!(dθ, θ, param, t)

    u1 = input_obc(t,param[1:3])
    u2 = input_obc(t,param[4:6])
    u3 = input_obc(t,param[7:9])
    
    u_in = [u1, u2, u3]
    diffusion!(dθ, θ, plate, property, boundary, actuation, u_in)
end



tspan = (0.0, Tf)
θ₀ = 300;
Ntotal = N₁*N₂
θinit = θ₀*ones(Ntotal)

dΘ = similar(θinit)

# pinit = repeat(p_fbc,3)
pinit = repeat(p_energy,3)

heat_conduction!(dΘ,θinit,pinit,0)

# Δt = 1e-2               # Sampling time

using OrdinaryDiffEq

t_samp = 10.0
alg = KenCarp5()
prob = ODEProblem(heat_conduction!,θinit,tspan,pinit)
sol = solve(prob,alg,p=pinit, saveat=t_samp)

using Plots
heatmap(reshape(Array(sol)[:,end],N₁,N₂)')

ref_data = repeat(ref.(sol.t)',3)


### Sensor ###
num_sensor = 3        # Number of sensors
sensing = IOSetup(plate)
config_sensor  = RadialCharacteristics(1.0, 1, 0)
setIOSetup!(sensing, plate, num_sensor, config_sensor, :north)

C = zeros(num_sensor,Ntotal)
b_sym = :north
ids = unique(sensing.identifier[b_sym])
for i in ids
    idx = findall(x-> x==i, sensing.identifier[b_sym])
    boundary_idx = sensing.indices[b_sym]
    boundary_char = sensing.character[b_sym]
    C[i,boundary_idx[idx]] = boundary_char[idx] / sum(boundary_char[idx])
end

y = C*Array(sol)
err = ref_data - y
sum(abs2, (sol.t/Tf)' .* err) /Tf

function loss_optim_std(u,p)
    pars = u #vcat(u[1:2],p[1],u[3:4],p[2],u[5:6],p[3])
#    pars = [u[1], u[2], p[1]]
    sol_loss = solve(prob, alg,p=pars, saveat = t_samp)
    if sol_loss.retcode == ReturnCode.Success
        y = C*Array(sol_loss)
        err = ref_data - y
        err = (sol.t/Tf)' .* err
        loss = sum(abs2, err) / Tf
    else
        loss = Inf
    end
    return loss
end

const store_loss=[]
global store_param=[]

callback = function (state, l) 
    # store loss and parameters
    # append!(store_loss,l) # Loss values 
    
    # store_param must be global
    # global store_param  
    # store_param = vcat(store_param,[state.u]) # save with vcat


    println(l)
    #println("iter")

    if l < -0.2
        return true
    end

    return false
end



# opt_p = repeat([p_energy[3]],num_actuators)
# opt_u0 = repeat(p_energy[1:2],num_actuators)

p_opt = repeat(p_energy,3)

loss_optim_std(p_opt, [0])
loss_output(p_opt,0)

# vcat(opt_u0[1:2],opt_p[1],opt_u0[3:4],opt_p[2],opt_u0[5:6],opt_p[3])

using Optimization, OptimizationOptimJL, ForwardDiff
optf = OptimizationFunction(loss_optim_std, Optimization.AutoForwardDiff())
opt_prob = OptimizationProblem(optf, p_opt, [0])
p_final = Optimization.solve(opt_prob, ConjugateGradient(), callback=callback, maxiters=20)

# optf = OptimizationFunction(loss_output, Optimization.AutoForwardDiff())
# opt_prob = OptimizationProblem(optf, p_opt, lb=repeat([0.0, 1.0, 0.0],3), ub=repeat([20, Inf, Inf],3));
# p12 = Optimization.solve(opt_prob, ConjugateGradient(), callback=callback, maxiters=5)

# p_final = vcat(p12[1:2],opt_p[1],p12[3:4],opt_p[2],p12[5:6],opt_p[3])
# sol_final = solve(prob,alg,p=p_final, saveat=t_samp)

sol_final = solve(prob,alg,p=p_final, saveat=t_samp)


using Plots
heatmap(reshape(Array(sol_final)[:,end],N₁,N₂)')


loss_output(p12)

y_final = C*Array(sol_final)
err = ref_data - y
loss = sum(abs2, err) / Tf

using Plots
plot(sol_final)

plot(sol_final.t, y_final')
scatter!(sol_final.t[1:20:end], ref_data[:,1:20:end]')


plot(sol_final.t, err')