λapprox = [370, -2.85, 8.458e-3, -1e-5, 4.1667e-9]
using Hestia
prop_dynamic = DynamicIsotropic(λapprox, [8000], [400])

L = 0.2
N = 40
rod = HeatRod(L, N)
boundary = Boundary(rod)

function heat_conduction_dynamic!(dw, w, param, t)
    diffusion!(dw, w, rod, prop_dynamic, boundary)
end

using OrdinaryDiffEq
θinit = 300ones(N)
θinit[1:round(Int,N/2)] .= 700 
tspan = (0.0, 1200)
tsave = 20.0;
alg = KenCarp5()
prob_dynamic = ODEProblem(heat_conduction_dynamic!,θinit,tspan)
sol_dynamic = solve(prob_dynamic, alg, saveat=tsave)

sum(θinit)/N
temp_data = Array(sol_dynamic)

sum((500 .-sum(temp_data,dims=1)'/N))

using Plots
plot(sol_dynamic)

