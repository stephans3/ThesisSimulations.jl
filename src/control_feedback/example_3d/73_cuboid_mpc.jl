


θvec = collect(300:50:500)
M_temp = mapreduce(z-> [1  z  z^2 z^3 z^4], vcat, θvec)

λ1data = [40,44,50,52,52.5]
λ1p = inv(M_temp)*λ1data

λ3data = [40,55,60,65,68]
λ3p = inv(M_temp)*λ3data

ρ = 8000; # Density
c = 400;  # Specific heat capacity

L = 0.2; # Length 
W = 0.2; # Width
H = 0.05;# Height
N₁ = 10;
N₂ = 10;
N₃ = 5;
Nc = N₁*N₂*N₃ 
Δx₁ = L/N₁
Δx₂ = W/N₂
Δx₃ = W/N₃


using Hestia 

property = DynamicAnisotropic(λ1p, λ1p, λ3p, [ρ],[c])
cuboid  = HeatCuboid(L,W, H, N₁,N₂,N₃)
boundary = Boundary(cuboid)

### Boundaries ###
Θamb = 300.0;
h = 10;
ε = 0.1;
emission = Emission(h, Θamb,ε) #  Stefan-Boltzmann BC: heat transfer (linear) and heat radiation (quartic/nonlinear)
boundary = Boundary(cuboid)

setEmission!(boundary, emission, :east )
setEmission!(boundary, emission, :south )
setEmission!(boundary, emission, :topside )

### Actuation ###
actuation = IOSetup(cuboid)
num_actuators = (2,2)        # Number of actuators per boundary
Nu = num_actuators[1]* num_actuators[2]
config  = RadialCharacteristics(1.0, 2, 30)
# config  = RadialCharacteristics(1.0, 2, 0)
setIOSetup!(actuation, cuboid, num_actuators, config, :underside)


# Reference
ref_init = 300;
Δr = 200; # Difference operating points
ps = 10; # Steepness
ψ(t) = (1+tanh(10*(t/Tf - 1/2)))/2
ref(t) = ref_init + Δr*ψ(t)


function input_obc(t,p)
    return exp(p[1] - p[3]^2 * (t/T_ff - 1/p[2])^2)
end



function heat_conduction_heat_up!(dθ, θ, param, t)
    u1 = input_obc(t,param[1:3])
    u2 = input_obc(t,param[4:6])
    u3 = input_obc(t,param[7:9])
    u4 = input_obc(t,param[10:12])

    u_in = [u1, u2, u3, u4]
    diffusion!(dθ, θ, cuboid, property, boundary, actuation, u_in)
end

p_opt =
[12.920528591175692
  2.055521226797314
  7.907112254486979
 12.980761600310702
  2.0513665968206873
  8.426793438967957
 12.961190624776055
  2.0691384801460204
  8.562627674914927]


#=
# New:
p_opt =
[12.920528591175692
  2.055521226797314
  7.907112254486979
 12.980761600310702
  2.0513665968206873
  8.426793438967957
 12.961190624776055
  2.0691384801460204
  8.562627674914927]
=#

#=
# Old:
p_opt = [13.010951664331811
        1.9765036457312297
        8.160716631530311
        12.892699280408907
        2.153289753582373
        8.203973434811285
        12.911826304389534
        2.145243910846501
        8.299341977569908
        12.978783755664848
        1.9883521480910786
        8.285147497006298]
=#

p_ff = vcat(p_opt,p_opt[1:3])
T_ff = 1200;
tspan_ff = (0.0, T_ff)
θ₀ = 300;
θinit_ff = θ₀*ones(Nc)
dΘ = similar(θinit_ff)
heat_conduction_heat_up!(dΘ,θinit_ff,p_ff,0)

using OrdinaryDiffEq
t_samp = 30.0
alg = KenCarp5()
prob_ff = ODEProblem(heat_conduction_heat_up!,θinit_ff,tspan_ff,p_ff)
sol_ff = solve(prob_ff,alg,p=p_ff, saveat=t_samp)

using Plots
plot(sol_ff)



function input_mpc(p, t)
    if t <= 0
        return zeros(Nu)
    else
        n = ceil(Int64, t/t_mpc)
        nstart = Nu*(n-1) + 1
        nend = nstart + (Nu-1)
        return p[nstart:nend]
    end
end


# Heat Conduction Simulation
function heat_conduction_mpc!(dθ, θ, param, t)
    u_in = input_mpc(param, t)
    diffusion!(dθ, θ, cuboid, property, boundary, actuation, u_in)
end


t_mpc = 30;
N_mpc = 3;
T_stab = t_mpc*N_mpc;  #60 # 1200;
Tf = T_ff + T_stab;
tspan = (0,T_stab) # (T_ff, Tf)
#θ₀ = 500;
#Ntotal = N₁*N₂
θinit = Array(sol_ff[end]) #θ₀*ones(Ntotal)

#p_mpc_test = repeat(collect(1:4),N_mpc)
p_mpc_test = vcat(collect(1:4),collect(5:8),collect(9:12))
map(t-> input_mpc(p_mpc_test,t), 0:2:t_mpc)


dΘ = similar(θinit)
heat_conduction_mpc!(dΘ,θinit,ones(3),0)


# Δt = 1e-2               # Sampling time

using OrdinaryDiffEq

pinit = vcat(5e4*ones(Nu), 3e4*ones(Nu), 1e4*ones(Nu))

#5e4*ones(Nu*N_mpc) # vcat([21e3,22e3,23e3,24e3], [4e3,3e3,2e3,1e3],[6e3,7e3,8e3,9e3])

t_samp = 10.0
alg = KenCarp5()
prob_orig = ODEProblem(heat_conduction_mpc!,θinit,tspan)
sol_mpc_test = solve(prob_orig,alg,p=pinit, saveat=t_mpc)

plot(sol_mpc_test)


### Sensor ###
num_sensor = (2,2)        # Number of sensors
Ny = num_sensor[1]*num_sensor[2]
sensing = IOSetup(cuboid)
config_sensor  = RadialCharacteristics(1.0, 1, 0)
setIOSetup!(sensing, cuboid, num_sensor, config_sensor, :topside)


C = zeros(Ny,Nc)
b_sym = :topside
ids = unique(sensing.identifier[b_sym])
for i in ids
    idx = findall(x-> x==i, sensing.identifier[b_sym])
    boundary_idx = sensing.indices[b_sym]
    boundary_char = sensing.character[b_sym]
    C[i,boundary_idx[idx]] = boundary_char[idx] / sum(boundary_char[idx])
end

y_out = C*Array(sol_mpc_test)
ref_data = 500*ones(Ny);


function loss_optim(u,p,prob)
    prob = prob
    pars = exp.(u)
    sol_loss = solve(prob, alg,p=pars, saveat = t_mpc)
    if sol_loss.retcode == ReturnCode.Success
        y = C*Array(sol_loss)
        err = ref_data .- y
        loss_y = sum(abs2, err) / ((N_mpc+1)*Ny)
        pars_arr = reshape(pars, Nu, N_mpc)
        loss_u =  (1e-8)*sum(abs2, pars_arr[:,1:end-1] - pars_arr[:,2:end])/((N_mpc-1)*Nu)
        loss = loss_y + loss_u
    else
        loss = Inf
    end
    return loss
end

# t_mpc = 30; # MPC sampling time
# N_mpc = 3; # MPC control horizon

N_mpc_samples = 10
p_data_store = zeros(3,N_mpc_samples);

A_east = 1e-2  # W*H;
A_south = 1e-2 # L*H;
A_top = 4e-2   # L*W;

ΔΘ = 5; # Temperature offset
ΔP = c*ρ*ΔΘ*(L*W*H) / t_mpc
Pem_approx =   (A_east + A_south + A_top)*emit(500,emission) 

act_int = sum(actuation.character[:underside])*Δx₁*Δx₂
u_init_av1 = (abs(Pem_approx) + ΔP) / act_int
u_init_av2 = (abs(Pem_approx)) / act_int

pinit = vcat(log(u_init_av1)*ones(Nu),log(u_init_av2)*ones(Nu*(N_mpc-1)))
# pinit = log(u_init_av)*ones(Nu*N_mpc) 
# loss_optim(10*ones(Nu*N_mpc),[],prob_orig)

loss_optim(pinit,[],prob_orig)

using Optimization, OptimizationOptimJL, ForwardDiff
prob_ode = remake(prob_orig)
loss_fun(u,p)  = loss_optim(u,p,prob_ode)
optf = OptimizationFunction(loss_fun, Optimization.AutoForwardDiff())
opt_prob = OptimizationProblem(optf, pinit, [0])

n=1
p_opt = Optimization.solve(opt_prob, ConjugateGradient(), maxiters=3)
sol_temp = solve(prob_ode,alg,p=exp.(Array(p_opt)), save_everystep=false)

prob_ode = remake(prob_ode; u0 = sol_temp[:,end])
    p_opt_data = Array(p_opt)
    p_data_store[:,n] = p_opt_data[1:3]
    p_new = vcat(p_opt_data[4:9],p_opt_data[7:9])

    loss_fun(u,p) = loss_optim(u,p,prob_ode)
    optf = OptimizationFunction(loss_fun, Optimization.AutoForwardDiff())
    opt_prob = remake(opt_prob; f=optf, u0=p_new)



#=
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

p_opt_data = p_data_store
=#
