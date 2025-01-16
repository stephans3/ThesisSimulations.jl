
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


### Actuation ###
actuation = IOSetup(plate)
num_actuators = 3        # Number of actuators per boundary
config  = RadialCharacteristics(1.0, 2, 30)
setIOSetup!(actuation, plate, num_actuators, config, :south)

actuator_char = getCharacteristics(actuation, :south)[1]

### Sensor ###
num_sensor = 3        # Number of sensors
sensing = IOSetup(plate)
config_sensor  = RadialCharacteristics(1.0, 1, 0)
setIOSetup!(sensing, plate, num_sensor, config_sensor, :north)

sensor_char = getCharacteristics(sensing, :north)[1]
C = hcat(zeros(3,N₁*(N₂-1)), sensor_char' ./  sum(sensor_char, dims=1)')
# t_mpc = 30;
# N_mpc = 3;
# -> t_sim = 3*30 = 90
# t/t_sim



# p_in = [1 3 5; 2 4 6; 7 8 9]
# tgrid = 0:0.1:N_mpc*t_mpc
# map(t-> input(p_in,t),tgrid)

# Heat Conduction Simulation
function heat_conduction!(dθ, θ, param, t)
    u_in = [6,3,6]*1e3 #1e4*ones(3)
    diffusion!(dθ, θ, plate, property, boundary, actuation, u_in)
end


Tf = 60;
tspan = (0.0, Tf)
θ₀ = 500;
Ntotal = N₁*N₂
θinit = θ₀*ones(Ntotal)

# dΘ = similar(θinit)
# heat_conduction!(dΘ,θinit,ones(3),0)


# Δt = 1e-2               # Sampling time

using OrdinaryDiffEq

t_save = 1;
alg = KenCarp5()
prob_orig = ODEProblem(heat_conduction!,θinit,tspan)
sol = solve(prob_orig,alg, saveat=t_save)

yout = transpose(C*Array(sol))
tgrid = sol.t
#using Plots
#plot(sol.t, yout)



using CairoMakie
path2folder = "results/figures/controlled/feedback/"
begin
    filename = path2folder*"mpc_example_step_response.pdf"

    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Time in [s]", ylabel = "Temperature in [K]", xlabelsize = 30, ylabelsize = 30,
    xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1)#, limits = (nothing, (500, 500.75)))
    
    #ax1.xticks = collect(0:60:300)
    #ax1.yticks = collect(500:0.25:501)
    lines!(tgrid, yout[:,1], linestyle = :dot,  linewidth = 5, label="Sensor 1")
    lines!(tgrid, yout[:,2], linestyle = :dash,  linewidth = 5, label="Sensor 2")

    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f

    save(filename, f, pt_per_unit = 1)   
end