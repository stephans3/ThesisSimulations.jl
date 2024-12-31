
L = 0.05; # Length of 1D rod
# Aluminium
ρ = 8000; # Density
c = 400;  # Specific heat capacity

N₁ = 10;
Δx = L/N₁

θvec = collect(300:50:500)
M_temp = mapreduce(z-> [1  z  z^2 z^3 z^4], vcat, θvec)

λdata = [40,55,60,65,68]
λp = inv(M_temp)*λdata

using Hestia
prop = DynamicIsotropic(λp, [ρ], [c])

rod = HeatRod(L, N₁)
boundary = Boundary(rod)

θamb = 300.0;
h = 10;
Θamb = 300;
ε = 0.1;
emission = Emission(h, Θamb,ε) #  Stefan-Boltzmann BC: heat transfer (linear) and heat radiation (quartic/nonlinear)
setEmission!(boundary, emission, :west )

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


#=
p_found = [11.83256078184801
            2.1052631578947367
            9.38415971553497]
=#
using OrdinaryDiffEq
θinit =  300* ones(N₁) # Intial values
Tf    =  1200;
tspan =  (0.0, Tf)   # Time span
tsamp = 1.0;
p_orig= p_found
alg = KenCarp4()    # Numerical integrator
prob_orig = ODEProblem(heat_conduction!,θinit,tspan,p_orig)
sol_orig = solve(prob_orig, alg, saveat = tsamp)

using Plots
plot(sol_orig)