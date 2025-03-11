#=
    Energy-based Optimization of 3-dimensional heat conduction
=#

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
setEmission!(boundary, emission,  :topside )

### Actuation ###
actuation = IOSetup(cuboid)
num_actuators = (2,2)        # Number of actuators per boundary
config  = RadialCharacteristics(1.0, 2, 30)
# config  = RadialCharacteristics(1.0, 2, 0)
setIOSetup!(actuation, cuboid, num_actuators, config, :underside)


function input_obc(t,p)
    return exp(p[1] - p[3]^2 * (t/Tf - 1/p[2])^2)
end


# Reference
ref_init = 300;
Δr = 200; # Difference operating points
ps = 10; # Steepness
ψ(t) = (1+tanh(10*(t/Tf - 1/2)))/2
ref(t) = ref_init + Δr*ψ(t)

Tf = 1200;
ts = (Tf/1000)
tgrid = 0 : ts : Tf

p_fbc = [11.81522609265776
            2.070393374741201
            9.212088025161702]

# Volume of cuboid
Ωvol = 2e-3; # =L*W*H

# Internal energy
ΔU = c*ρ*Δr*Ωvol

# Emitted thermal energy
E_tr = (h*(ts*sum(ref.(tgrid[2:end])) - Tf*Θamb))
coeff_rad = ε*5.67*1e-8;
E_rad = (coeff_rad*ts*sum(ref.(tgrid[2:end]).^4))

A_east = 1e-2  # W*H;
A_south = 1e-2 # L*H;
A_top = 4e-2   # L*W;

E_em_approx =  (A_east + A_south + A_top)*(E_tr + E_rad)

using SpecialFunctions
u_in_energy(p₁,p₂,p₃) = exp(p₁)*Tf*sqrt(pi) * (erf(p₃-p₃/p₂) - erf(-p₃/p₂)) / (2 * p₃)
E_oc_fbc_approx = u_in_energy(p_fbc...)


act_char_int = (sum(actuation.character[:underside])*Δx₁*Δx₂)
E_in_fbc_approx = E_oc_fbc_approx*act_char_int

E_needed = ΔU +E_em_approx
E_necessary_perc = 100*E_in_fbc_approx / E_needed


function loss_energy(u,p)
    return (E_em_approx + ΔU - act_char_int*u_in_energy(u[1],p[1],u[2]))^2 / Tf
end



loss_energy([p_fbc[1],p_fbc[3]],p_fbc[2])

const store_loss=[]
global store_param=[]

callback = function (state, l) 
    # store loss and parameters
    append!(store_loss,l) # Loss values 
    
    # store_param must be global
    global store_param  
    store_param = vcat(store_param,[state.u]) # save with vcat

    #println("iter")

    return false
end


# Find optimal p3
using Optimization, OptimizationOptimJL, ForwardDiff
opt_p = [p_fbc[2]]
opt_u0 = [p_fbc[1],p_fbc[3]]
loss_energy(opt_u0, opt_p)

optf = OptimizationFunction(loss_energy, Optimization.AutoForwardDiff())
opt_prob = OptimizationProblem(optf, opt_u0, opt_p)
p13 = Optimization.solve(opt_prob, ConjugateGradient(), callback=callback, maxiters=100)

p_energy = [p13[1], p_fbc[2],p13[2]]

store_loss
store_param



function heat_conduction!(dθ, θ, param, t)
    u1 = input_obc(t,param)
    u_in = u1*ones(4);
    diffusion!(dθ, θ, cuboid, property, boundary, actuation, u_in)
end


p_energy = [13.065247340633551
             2.070393374741201
             9.076210596231089]

pinit = p_energy
Tf = 1200;
tspan = (0.0, Tf)
θ₀ = 300;
θinit = θ₀*ones(Nc)
dΘ = similar(θinit)
heat_conduction!(dΘ,θinit,pinit,0)




# Δt = 1e-2               # Sampling time

using OrdinaryDiffEq

t_samp = 10.0
alg = KenCarp5()
prob = ODEProblem(heat_conduction!,θinit,tspan,pinit)
sol = solve(prob,alg,p=pinit, saveat=t_samp)


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
yout = C*Array(sol)

ref_data = repeat(ref.(sol.t)',Ny)





using CairoMakie
path2folder = "results/figures/controlled/cuboid_example/"
begin

    filename = path2folder*"cuboid_energy_output_1.pdf"

    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Time in [s]", ylabel = L"Output in [K] $~$", xlabelsize = 30, ylabelsize = 30,
    xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,) #, limits = (nothing, (499.2, 500.05))
    
    tgrid = sol.t[1:end]
    lines!(tgrid, yout[1,:], linestyle = :solid,  linewidth = 5, label="Sensor 1")
    lines!(tgrid, yout[2,:], linestyle = :dash,  linewidth = 5, label="Sensor 2")
    lines!(tgrid, yout[3,:], linestyle = :dot,  linewidth = 5, label="Sensor 3")
    lines!(tgrid, yout[4,:], linestyle = :dashdot,  linewidth = 5, label="Sensor 4")
    scatter!(tgrid[1:15:end], ref_data[1,1:15:end];  markersize = 15, marker = :diamond, color=:black, label = "Reference")
    ax1.xticks = collect(0:200:1200)
#    ax1.yticks = [499.2, 499.4, 499.6, 499.8, 500];
    axislegend(; position = :rb, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f
    
    # save(filename, f, pt_per_unit = 1)   
end

begin

    filename = path2folder*"cuboid_energy_output_2.pdf"

    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Time in [s]", ylabel = L"Output in [K] $~$", xlabelsize = 30, ylabelsize = 30,
    xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1, limits = (nothing, (490, 510)))
    
    tstart = 76;
    tgrid = sol.t[tstart:end]
    lines!(tgrid, yout[1,tstart:end], linestyle = :solid,  linewidth = 5, label="Sensor 1")
    lines!(tgrid, yout[2,tstart:end], linestyle = :dash,  linewidth = 5, label="Sensor 2")
    lines!(tgrid, yout[3,tstart:end], linestyle = :dot,  linewidth = 5, label="Sensor 3")
    lines!(tgrid, yout[4,tstart:end], linestyle = :dashdot,  linewidth = 5, label="Sensor 4")
    scatter!(sol.t[81:3:end], ref_data[1,81:3:end];  markersize = 15, marker = :diamond, color=:black, label = "Reference")

    ax1.xticks = collect(0:100:1200)
#    ax1.yticks = [499.2, 499.4, 499.6, 499.8, 500];
    #axislegend(; position = :rb, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f
    
    save(filename, f, pt_per_unit = 1)   
end


sol_data = reshape(Array(sol[end]),N₁,N₂,N₃)
x1grid = L/(2N₁) : Δx₁ : L
x2grid = W/(2N₂) : Δx₂ : W
x3grid = H/(2N₃) : Δx₃ : H