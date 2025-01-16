
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

#=
α = λ₂/(ρ*c)
t_diff = (W^2)/α
=#


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

u_init_av = (942 / (3*sum(actuator_char[:,1])*Δx₁));
pinit = log(u_init_av)*ones(3*N_mpc) # 8*ones(3*N_mpc)

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

p_opt_data = p_data_store

#=
# pinit = 8*ones(3*N_mpc)
p_opt_data = [
9.51674  8.69653  8.88635  8.63213  8.83793  8.74208  8.79865  8.75205  8.79137  8.77228
 9.18024  7.97067  7.57852  7.59994  7.82952  7.81358  7.83333  7.82207  7.84787  7.84244
 9.51674  8.69653  8.88635  8.63213  8.83793  8.74208  8.79865  8.75205  8.79137  8.77228]
=#

#=
# u_init_av = (942 / (3*sum(actuator_char[:,1])*Δx₁));
# pinit = log(u_init_av)*ones(3*N_mpc)
  p_opt_data = [9.30711  9.00171  8.64495  8.84113  8.7703   8.77578  8.77755  8.77696  8.77679  8.7753
     9.07901  7.75712  7.34858  7.6263   7.82564  7.82881  7.83985  7.85068  7.85308  7.85702
     9.30711  9.00171  8.64495  8.84113  8.7703   8.77578  8.77755  8.77696  8.77679  8.7753]
=#


#exp.(p_data_store)
p_store_vec = exp.(reshape(p_data_store,3*N_mpc_samples))


Tf = t_mpc*N_mpc_samples; 
tspan = (0.0, Tf)
θ₀ = 500;
θinit = θ₀*ones(Ntotal)

t_samp = 1.0
alg = KenCarp5()
prob_demo = ODEProblem(heat_conduction!,θinit,tspan)
sol_demo = solve(prob_demo,alg,p=p_store_vec, saveat=t_samp)
θsol = Array(sol_demo)


n_rep = round(Int64,t_mpc / t_samp)
u1_in = mapreduce(p -> exp(p)*ones(n_rep),vcat, p_data_store[1,:])
u2_in = mapreduce(p -> exp(p)*ones(n_rep),vcat, p_data_store[2,:])
udata = hcat(u1_in,u2_in,u1_in)


using CairoMakie
path2folder = "results/figures/controlled/feedback/"
begin
    filename = path2folder*"mpc_input.pdf"

    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Time in [s]", ylabel = L"Input $\times 10^{3}$", xlabelsize = 30, ylabelsize = 30,
    xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1)
    
    ax1.xticks = collect(0:60:300)
    scale = 1e-3;
    tgrid = sol_demo.t[1:end-1]
    lines!(tgrid, scale*udata[:,1], linestyle = :solid,  linewidth = 5, label="Actuator 1")
    lines!(tgrid, scale*udata[:,2], linestyle = :dash,  linewidth = 5, label="Actuator 2")

    axislegend(; position = :rt, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f

    save(filename, f, pt_per_unit = 1)   
end


begin
    yout = C*θsol
    filename = path2folder*"mpc_output.pdf"

    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Time in [s]", ylabel = L"Output in [K] $~$", xlabelsize = 30, ylabelsize = 30, limits = (nothing, (499.2, 500.05)),
    xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,)
    
    scale = 1;
    tgrid = sol_demo.t[1:end]
    lines!(tgrid, scale*yout[1,:], linestyle = :dot,  linewidth = 5, label="Sensor 1")
    lines!(tgrid, scale*yout[2,:], linestyle = :dash,  linewidth = 5, label="Sensor 2")

    ax1.xticks = collect(0:60:300)
    ax1.yticks = [499.2, 499.4, 499.6, 499.8, 500];
    axislegend(; position = :rb, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f
    
    save(filename, f, pt_per_unit = 1)   
end


# Emitted Power

n_ts = length(sol_demo.t)
θsol_2d = reshape(θsol, N₁,N₂,n_ts)
ϕem_W = map(θ-> emit(θ,emission), θsol_2d[1,1:N₂,:])
ϕem_E = map(θ-> emit(θ,emission), θsol_2d[N₁,1:N₂,:])
ϕem_N = map(θ-> emit(θ,emission), θsol_2d[1:N₁,N₂,:])

Pem_W = Δx₂*sum(ϕem_W,dims=1)[:]
Pem_E = Δx₂*sum(ϕem_E,dims=1)[:]
Pem_N = Δx₁*sum(ϕem_N,dims=1)[:]

Pem = Pem_W + Pem_E + Pem_N

# Approx. emitted power
# Pem_approx = L*emit(500,emission) + 2*W*emit(500,emission)

udata = hcat(u1_in,u2_in,u1_in)


# Supplied power
Pin = Δx₁*sum(vcat(actuator_char, zeros(N₁*(N₂-1),num_actuators))*udata',dims=1)[:]


begin
    filename = path2folder*"mpc_power.pdf"

    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Time in [s]", ylabel = L"Power in [W] $~$", xlabelsize = 30, ylabelsize = 30, limits = (nothing, (750, 2100)),
    xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,)
    
    scale = 1;
    tgrid = sol_demo.t[1:end-1]
    lines!(tgrid, scale*abs.(Pem[1:end-1]), linestyle = :solid,  linewidth = 5, label="Abs. Emitted")
    lines!(tgrid, scale*Pin, linestyle = :dash,  linewidth = 5, label="Supplied")

    ax1.yticks = collect(750:250:2000) # [0, 200, 400, 600, 800, 1000];
    ax1.xticks = collect(0:60:300)
    # lines!(num_iterations, pars_data[:,1], linestyle = :dash,  linewidth = 5)
    # lines!(num_iterations, pars_data[:,2], linestyle = :dash,  linewidth = 5)
    axislegend(; position = :rt, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f

    save(filename, f, pt_per_unit = 1)   
end


x1grid = Δx₁/2 : Δx₁ : L
x2grid = Δx₂/2 : Δx₂ : W
begin
    data = θsol_2d[:,:,end]
    filename = path2folder*"mpc_temp_contour.pdf"
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = L"Length $x_{1}$ in [cm]", ylabel = L"Width $x_{2}$ in [cm]", xlabelsize = 30, ylabelsize = 30)
    tightlimits!(ax1)
    # hidedecorations!(ax1)
    x1grid_cm = 100*x1grid
    x2grid_cm = 100*x2grid
    co = contourf!(ax1, x1grid_cm, x2grid_cm, data, colormap=:plasma, levels = 498:0.5:504) #levels = range(0.0, 10.0, length = 20))
    ax1.xticks = 0 : 5 : 30;
    ax1.yticks = 0 : 1 : 5;
    Colorbar(f[1, 2], co)
    #Colorbar(f[1, 2], co ,  ticks = [300, 301, 302])
    f    

    save(filename, f, pt_per_unit = 1)   
end

