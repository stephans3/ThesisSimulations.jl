using Hestia
prop = StaticIsotropic(50, 8000, 400)

L = 0.2;
W = 0.2;
H = 0.05;
N = (20, 20, 10)
cuboid = HeatCuboid(L, W, H, N...)

boundary = Boundary(cuboid)
h = 10;
Θamb = 300;
ε = 0.2;
emission = Emission(h, Θamb,ε) # Pure heat radiation
setEmission!(boundary, emission, :west);
setEmission!(boundary, emission, :south);
setEmission!(boundary, emission, :topside);

function heat_conduction!(dw, w, param, t)
    diffusion!(dw, w, cuboid, prop, boundary)
end

using OrdinaryDiffEq
θinit = 400ones(N[1]*N[2]*N[3]) 
tspan = (0.0, 10000)
tsave = 100.0;
alg = KenCarp5()
prob = ODEProblem(heat_conduction!,θinit,tspan)
sol = solve(prob,alg, saveat=tsave)


using Plots
plot(sol[1,:],legend=false)