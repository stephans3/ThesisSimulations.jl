using Hestia
prop_static1 = StaticIsotropic(40, 8000, 400)
prop_static2 = StaticIsotropic(60, 8000, 400)
prop_dynamic = DynamicIsotropic([10, 0.1], [8000], [400])

L = 0.2
N = 40
rod = HeatRod(L, N)

boundary = Boundary(rod)

function heat_conduction_static1!(dw, w, param, t)
    diffusion!(dw, w, rod, prop_static1, boundary)
end

function heat_conduction_static2!(dw, w, param, t)
    diffusion!(dw, w, rod, prop_static2, boundary)
end


function heat_conduction_dynamic!(dw, w, param, t)
    diffusion!(dw, w, rod, prop_dynamic, boundary)
end


using OrdinaryDiffEq
θinit = 300ones(N)
θinit[1:round(Int,N/2)] .= 500 
tspan = (0.0, 3000)
tsave = 100.0;
alg = KenCarp5()
prob_static1 = ODEProblem(heat_conduction_static1!,θinit,tspan)
prob_static2 = ODEProblem(heat_conduction_static2!,θinit,tspan)
prob_dynamic = ODEProblem(heat_conduction_dynamic!,θinit,tspan)

sol_static1 = solve(prob_static1, alg, saveat=tsave)
sol_static2 = solve(prob_static2, alg, saveat=tsave)
sol_dynamic = solve(prob_dynamic, alg, saveat=tsave)

using Plots
plot(sol_static1[end])
plot!(sol_static2[end])
plot!(sol_dynamic[end])

plot(sol_static1[2])
plot!(sol_static2[2])
plot!(sol_dynamic[2])

plot(sol_static1[10])
plot!(sol_static2[10])
plot!(sol_dynamic[10])