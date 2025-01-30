


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
Δx₃ = H/N₃


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

p_opt=
[12.894929657082693
  2.0554514321527058
  7.682015031653755
 12.967392692355489
  2.0538122351888752
  8.300509555164869
 12.943662543893003
  2.066455923317032
  8.459445553931738]


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

t_samp = 10.0
alg = KenCarp5()
prob_orig = ODEProblem(heat_conduction_mpc!,θinit,tspan)
sol_mpc_test = solve(prob_orig,alg,p=pinit, saveat=t_mpc)

# plot(sol_mpc_test)


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
p_data_store = zeros(Nu,N_mpc_samples);

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
# pinit[Nu*(N_mpc-1)+1:end]

using Optimization, OptimizationOptimJL, ForwardDiff
prob_ode = remake(prob_orig)
loss_fun(u,p)  = loss_optim(u,p,prob_ode)
optf = OptimizationFunction(loss_fun, Optimization.AutoForwardDiff())
opt_prob = OptimizationProblem(optf, pinit, [0])

#=
for n=1:N_mpc_samples
    p_opt = Optimization.solve(opt_prob, ConjugateGradient(), maxiters=3)
    sol_temp = solve(prob_ode,alg,p=exp.(Array(p_opt)), save_everystep=false)
    prob_ode = remake(prob_ode; u0 = sol_temp[:,end])
    p_opt_data = Array(p_opt)
    p_data_store[:,n] = p_opt_data[1:Nu]
    p_new = vcat(p_opt_data[Nu+1:end],pinit[Nu*(N_mpc-1)+1:end])

    loss_fun(u,p) = loss_optim(u,p,prob_ode)
    optf = OptimizationFunction(loss_fun, Optimization.AutoForwardDiff())
    opt_prob = remake(opt_prob; f=optf, u0=p_new)
end

p_opt_data = p_data_store
=#

#=
p_opt_data=
[10.8529  9.47944  9.07436  9.42223  9.43506  9.40945   9.31975  9.21962  9.30498  9.31311
11.5205  9.37013  9.57602  9.68577  9.77107  9.75241  10.1366   9.5841   9.77754  9.67058
10.1589  9.56055  8.15368  8.53608  8.78853  8.93208   8.73112  8.51229  8.65838  8.79611
10.8314  9.48784  9.0854   9.42571  9.43646  9.40954   9.31809  9.21926  9.30504  9.31356]
=#


p_opt_data=
[10.8529  9.47944  9.07436  9.42223  9.43506  9.40945   9.31975  9.21962  9.30498  9.31311
11.5205  9.37013  9.57602  9.68577  9.77107  9.75241  10.1366   9.5841   9.77754  9.67058
10.1589  9.56055  8.15368  8.53608  8.78853  8.93208   8.73112  8.51229  8.65838  8.79611
10.8314  9.48784  9.0854   9.42571  9.43646  9.40954   9.31809  9.21926  9.30504  9.31356]


p_store_vec = exp.(reshape(p_opt_data,Nu*N_mpc_samples))
Tf = t_mpc*N_mpc_samples; 
tspan = (0.0, Tf)
θinit = prob_orig.u0

t_samp = 1.0
alg = KenCarp5()
prob_demo = ODEProblem(heat_conduction_mpc!,θinit,tspan)
sol_demo = solve(prob_demo,alg,p=p_store_vec, saveat=t_samp)
θsol = Array(sol_demo)



using DelimitedFiles;
path2folder_export = "results/data/"
filename = "example_cuboid_mpc.csv"
path2file_export = path2folder_export * filename

data_export = Array(sol_demo)
open(path2file_export, "w") do io
    writedlm(io, data_export, ',')
end;


n_rep = round(Int64,t_mpc / t_samp)
u1_in = mapreduce(p -> exp(p)*ones(n_rep),vcat, p_opt_data[1,:])
u2_in = mapreduce(p -> exp(p)*ones(n_rep),vcat, p_opt_data[2,:])
u3_in = mapreduce(p -> exp(p)*ones(n_rep),vcat, p_opt_data[3,:])
u4_in = mapreduce(p -> exp(p)*ones(n_rep),vcat, p_opt_data[4,:])
udata = hcat(u1_in,u2_in,u3_in,u4_in)



using CairoMakie
path2folder = "results/figures/controlled/cuboid_example/"
begin
    filename = path2folder*"cuboid_mpc_input.pdf"

    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Time in [s]", ylabel = L"Input $\times 10^{3}$", xlabelsize = 30, ylabelsize = 30,
    xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1, yminorgridvisible = true,)
    
    ax1.xticks = collect(0:60:300)
    ax1.yticks = collect(0:20:100);
    scale = 1e-3;
    tgrid = sol_demo.t[1:end-1]
    lines!(tgrid, scale*udata[:,1], linestyle = :solid,  linewidth = 5, label="Actuator 1")
    lines!(tgrid, scale*udata[:,2], linestyle = :dash,  linewidth = 5, label="Actuator 2")
    lines!(tgrid, scale*udata[:,3], linestyle = :dot,  linewidth = 5, label="Actuator 3")
    lines!(tgrid, scale*udata[:,4], linestyle = :dashdot,  linewidth = 5, label="Actuator 4")

    axislegend(; position = :rt, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f

    save(filename, f, pt_per_unit = 1)   
end


begin
    yout = C*θsol
    filename = path2folder*"cuboid_mpc_output.pdf"

    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Time in [s]", ylabel = L"Output in [K] $~$", xlabelsize = 30, ylabelsize = 30,
    xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,limits = (nothing, (493, 498))) #, 
    
    scale = 1;
    tgrid = sol_demo.t[1:end]
    lines!(tgrid, scale*yout[1,:], linestyle = :solid,  linewidth = 5, label="Sensor 1")
    lines!(tgrid, scale*yout[2,:], linestyle = :dash,  linewidth = 5, label="Sensor 2")
    lines!(tgrid, scale*yout[3,:], linestyle = :dot,  linewidth = 5, label="Sensor 3")
    lines!(tgrid, scale*yout[4,:], linestyle = :dashdot,  linewidth = 5, label="Sensor 4")

    ax1.xticks = collect(0:60:300)
    #
    axislegend(; position = :rb, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f
    
    save(filename, f, pt_per_unit = 1)   
end


# Emitted Power

n_ts = length(sol_demo.t)
θsol_2d = reshape(θsol, N₁,N₂,N₃,n_ts)
ϕem_E = map(θ-> emit(θ,emission), θsol_2d[N₁,1:N₂,1:N₃,:])
ϕem_S = map(θ-> emit(θ,emission), θsol_2d[1:N₁,1,1:N₃,:])
ϕem_T = map(θ-> emit(θ,emission), θsol_2d[1:N₁,1:N₂,N₃,:])

Pem_E = Δx₂*Δx₃*sum(ϕem_E,dims=(1,2))[:]
Pem_S = Δx₁*Δx₃*sum(ϕem_S,dims=(1,2))[:]
Pem_T = Δx₁*Δx₂*sum(ϕem_T,dims=(1,2))[:]

Pem = Pem_E + Pem_S + Pem_T


# Supplied power
act_char = reshape(actuation.character[:underside], 5, 5, 4)
Pin_per_act = Δx₁*Δx₂*mapreduce( n-> sum(act_char[:,:,n])*udata[:,n],hcat,1:Nu)
Pin_total = sum(Pin_per_act,dims=2)[:]



begin
    filename = path2folder*"cuboid_mpc_power.pdf"

    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Time in [s]", ylabel = L"Power in [W] $~$", xlabelsize = 30, ylabelsize = 30,
    xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,) #
    
    scale = 1;
    tgrid = sol_demo.t[1:end-1]
    lines!(tgrid, scale*abs.(Pem[1:end-1]), linestyle = :solid,  linewidth = 5, label="Abs. Emitted")
    lines!(tgrid, scale*Pin_total, linestyle = :dash,  linewidth = 5, label="Supplied")

    ax1.yticks = collect(100:100:700) # [0, 200, 400, 600, 800, 1000];
    ax1.xticks = collect(0:60:300)
    # lines!(num_iterations, pars_data[:,1], linestyle = :dash,  linewidth = 5)
    # lines!(num_iterations, pars_data[:,2], linestyle = :dash,  linewidth = 5)
    axislegend(; position = :rt, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f

    save(filename, f, pt_per_unit = 1)   
end


x1grid = Δx₁/2 : Δx₁ : L
x2grid = Δx₂/2 : Δx₂ : W
x3grid = Δx₃/2 : Δx₃ : H


begin
    data = θsol_2d[:,:,1,end]
    filename = path2folder*"cuboid_mpc_bottomview.pdf"
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = L"Length $x_{1}$ in [cm]", ylabel = L"Width $x_{2}$ in [cm]", xlabelsize = 30, ylabelsize = 30)
    tightlimits!(ax1)
    # hidedecorations!(ax1)
    x1grid_cm = 100*x1grid
    x2grid_cm = 100*x2grid
    co = contourf!(ax1, x1grid_cm, x2grid_cm, data, colormap=:plasma, levels = 495:0.5:503) #levels = range(0.0, 10.0, length = 20))
    #ax1.xticks = 0 : 5 : 30;
    #ax1.yticks = 0 : 1 : 5;
    Colorbar(f[1, 2], co ,  ticks = collect(495:2:503))
    f    

    save(filename, f, pt_per_unit = 1)   
end

begin
    data = θsol_2d[:,:,end,end]
    filename = path2folder*"cuboid_mpc_topview.pdf"
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = L"Length $x_{1}$ in [cm]", ylabel = L"Width $x_{2}$ in [cm]", xlabelsize = 30, ylabelsize = 30)
    tightlimits!(ax1)
    # hidedecorations!(ax1)
    x1grid_cm = 100*x1grid
    x2grid_cm = 100*x2grid
    co = contourf!(ax1, x1grid_cm, x2grid_cm, data, colormap=:plasma, levels = 495:0.5:503) #levels = range(0.0, 10.0, length = 20))
    #ax1.xticks = 0 : 5 : 30;
    #ax1.yticks = 0 : 1 : 5;
    Colorbar(f[1, 2], co ,  ticks = collect(495:2:503))
    f    

    save(filename, f, pt_per_unit = 1)   
end


begin
    data = θsol_2d[:,1,:,end]
    filename = path2folder*"cuboid_mpc_side_south.pdf"
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = L"Length $x_{1}$ in [cm]", ylabel = L"Height $x_{3}$ in [cm]", xlabelsize = 30, ylabelsize = 30)
    tightlimits!(ax1)
    # hidedecorations!(ax1)
    x1grid_cm = 100*x1grid
    x3grid_cm = 100*x3grid
    co = contourf!(ax1, x1grid_cm, x3grid_cm, data, colormap=:plasma, levels = 495:0.5:503) #levels = range(0.0, 10.0, length = 20))
    #ax1.xticks = 0 : 5 : 30;
    ax1.yticks = 1 : 5;
    Colorbar(f[1, 2], co ,  ticks = collect(495:2:503))
    f    

    save(filename, f, pt_per_unit = 1)   
end



begin
    data = θsol_2d[end,:,:,end]
    filename = path2folder*"cuboid_mpc_side_east.pdf"
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = L"Width $x_{2}$ in [cm]", ylabel = L"Height $x_{3}$ in [cm]", xlabelsize = 30, ylabelsize = 30)
    tightlimits!(ax1)
    # hidedecorations!(ax1)
    x2grid_cm = 100*x2grid
    x3grid_cm = 100*x3grid
    co = contourf!(ax1, x2grid_cm, x3grid_cm, data, colormap=:plasma, levels = 495:0.5:503) #levels = range(0.0, 10.0, length = 20))
    #ax1.xticks = 0 : 5 : 30;
    ax1.yticks = 1 : 5;
    Colorbar(f[1, 2], co ,  ticks = collect(495:2:503))
    f    

    save(filename, f, pt_per_unit = 1)   
end

