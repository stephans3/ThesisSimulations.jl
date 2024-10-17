using Hestia
property = StaticIsotropic(1e-5, 1, 1)

L = 0.1
N = 20;
heatrod  = HeatRod(L, N)

θamb = 300.0;
boundary_transfer = Boundary(heatrod)
em_transfer = Emission(10, 0, θamb)
setEmission!(boundary_transfer, em_transfer, :east);

boundary_radiation = Boundary(heatrod)
em_radiation = Emission(0, 0.5, 0)
setEmission!(boundary_radiation, em_radiation, :east);

boundary_complete = Boundary(heatrod)
em_complete = Emission(10, 0.5, θamb)
setEmission!(boundary_complete, em_complete, :east);

### Simulation ###
function heat_conduction!(dw, w, param, t)
    dΘ1 = @views dw[1:N]
 #   dΘ2 = @views dw[N+1:2N]
 #   dΘ3 = @views dw[2N+1:3N]

    Θ1 = @views w[1:N]
 #   Θ2 = @views w[N+1:2N]
 #   Θ3 = @views w[2N+1:3N]

    diffusion!(dΘ1, Θ1, heatrod, property, boundary_transfer)
 #   diffusion!(dΘ2, Θ2, heatrod, property, boundary_radiation)
 #   diffusion!(dΘ3, Θ3, heatrod, property, boundary_complete)
end

θinit = 400ones(N)

using OrdinaryDiffEq
tspan = (0, 20.0)
alg = KenCarp5()
#prob= ODEProblem(heat_conduction!,vcat(θinit,θinit,θinit),tspan)
prob= ODEProblem(heat_conduction!,θinit,tspan)
sol = solve(prob,alg, saveat=1.0)