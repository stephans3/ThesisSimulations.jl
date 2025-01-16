
λ = 1;
ρ = 1; # Density
c = 1;  # Specific heat capacity

L = 1
N = 100;
Δx₁ = L/N


α = λ/(ρ*c)
t_diff = (L^2)/α



using Hestia 
property = StaticIsotropic(λ, ρ,c)
rod  = HeatRod(L,N)
boundary = Boundary(rod)

### Boundaries ###
#  Stefan-Boltzmann BC: heat transfer (linear) and heat radiation (quartic/nonlinear)
boundary = Boundary(rod)


### Actuation ###
actuation = IOSetup(rod)
num_actuators = 1        # Number of actuators per boundary
setIOSetup!(actuation, rod, 1,1, :west)

actuator_char = getCharacteristics(actuation, :west)[1]


# t_mpc = 30;
# N_mpc = 3;
# -> t_sim = 3*30 = 90
# t/t_sim


# p_in = [1 3 5; 2 4 6; 7 8 9]
# tgrid = 0:0.1:N_mpc*t_mpc
# map(t-> input(p_in,t),tgrid)

# Heat Conduction Simulation
function heat_conduction!(dθ, θ, param, t)
    u_in = 1e2*ones(1)
    diffusion!(dθ, θ, rod, property, boundary, actuation, u_in)
end


Tf = 2;  #60 # 1200;
tspan = (0.0, Tf)
θ₀ = 0;
θinit = θ₀*ones(N)
dΘ = similar(θinit)

heat_conduction!(dΘ, θinit, 0, 0)

# Δt = 1e-2               # Sampling time

using OrdinaryDiffEq

t_samp = 0.01
alg = KenCarp5()
prob_orig = ODEProblem(heat_conduction!,θinit,tspan)
sol = solve(prob_orig, saveat=t_samp)

temp_data = Array(sol)

using Plots
plot(sol.t, temp_data[end,:])
#temp_data[end,:]