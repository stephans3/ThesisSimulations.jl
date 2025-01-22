
λapprox = [10, 0.1]
using Hestia
prop_dynamic = DynamicIsotropic(λapprox, [8000], [400])
prop_static1 = StaticIsotropic(40, 8000, 400)
prop_static2 = StaticIsotropic(60, 8000, 400)

L = 0.1
N = 5
rod = HeatRod(L, N)
boundary = Boundary(rod)

function heat_conduction_dynamic!(dw, w, param, t)
    diffusion!(dw, w, rod, prop_dynamic, boundary)
end

function heat_conduction_static_low!(dw, w, param, t)
    diffusion!(dw, w, rod, prop_static1, boundary)
end

function heat_conduction_static_high!(dw, w, param, t)
    diffusion!(dw, w, rod, prop_static2, boundary)
end


using OrdinaryDiffEq
θinit = 300ones(N)
θinit[1:round(Int,N/2)] .= 500 
tspan = (0.0, 600)
tsave = 2.0;
alg = KenCarp5()

# Temperature-dependent
prob_dynamic = ODEProblem(heat_conduction_dynamic!,θinit,tspan)
sol_dynamic = solve(prob_dynamic, alg)#, saveat=tsave)

# Constant low
prob_static_low = ODEProblem(heat_conduction_static_low!,θinit,tspan)
sol_static_low = solve(prob_static_low, alg)#, saveat=tsave)

# Constant high
prob_static_high = ODEProblem(heat_conduction_static_high!,θinit,tspan)
sol_static_high = solve(prob_static_high, alg)#, saveat=tsave)


Θ1_dyn = Array(sol_dynamic)[1,:]
Θ1_st_low = Array(sol_static_low)[1,:]
Θ1_st_high = Array(sol_static_high)[1,:]

using Plots
plot(sol_dynamic.t, Θ1_dyn)
plot!(sol_static_low.t,Θ1_st_low)
plot!(sol_static_high.t,Θ1_st_high)


dΘ1_dyn = hcat([sol_dynamic.k[i][1] for i in 1:length(sol_dynamic)]...)[1,2:end]
dΘ1_st_low = hcat([sol_static_low.k[i][1] for i in 1:length(sol_static_low)]...)[1,2:end]
dΘ1_st_high = hcat([sol_static_high.k[i][1] for i in 1:length(sol_static_high)]...)[1,2:end]


plot(sol_dynamic.t[2:end], dΘ1_dyn)
plot!(sol_static_low.t[2:end],dΘ1_st_low)
plot!(sol_static_high.t[2:end],dΘ1_st_high)

Array(sol_dynamic.k)