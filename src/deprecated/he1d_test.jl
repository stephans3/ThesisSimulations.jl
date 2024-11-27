
L = 0.05; # Length of 1D rod
# Aluminium
λ = 60;  # Thermal conductivity
ρ = 8000; # Density
c = 400;  # Specific heat capacity
α = λ / (ρ * c) # Diffusivity

N₁ = 20;
Δx = L/N₁

using Hestia
prop = StaticIsotropic(λ, ρ, c)

rod = HeatRod(L, N₁)
boundary = Boundary(rod)

#=
h = 10;
Θamb = 300;
ε = 0.2;
em_total = Emission(h, Θamb,ε) 
setEmission!(boundary, em_total, :west);
setEmission!(boundary, em_total, :east);
=#

actuation = IOSetup(rod)
setIOSetup!(actuation, rod, 1, 1.0,  :west)

function input_obc(t,p)
    return exp(p[1] - p[3]^2 * (t/Tf - 1/p[2])^2)
end


function heat_conduction!(dw, w, param, t) 
    u_in1 = ones(1)*input_obc(t,param)

    diffusion!(dw, w, rod, prop, boundary,actuation,u_in1)
end


p_found = [11.81522609265776
2.070393374741201
9.212088025161702]

using OrdinaryDiffEq
θinit =  300* ones(N₁) # Intial values
Tf    =  1200;
tspan =  (0.0, Tf)   # Time span
tsamp = 1.0;
p_orig= p_found
alg = KenCarp4()    # Numerical integrator
prob_orig = ODEProblem(heat_conduction!,θinit,tspan,p_orig)
sol_orig = solve(prob_orig, alg, saveat = tsamp)