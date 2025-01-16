using Hestia
prop = StaticIsotropic(50, 8000, 400)

L = 0.2;
W = 0.2;
N = (20, 20)
plate = HeatPlate(L, W, N...)

boundary = Boundary(plate)
h = 20;
Θamb = 300;
ε = 0#0.2;
emission = Emission(h, Θamb,ε) # Pure heat radiation
setEmission!(boundary, emission, :west);
setEmission!(boundary, emission, :south);

function heat_conduction!(dw, w, param, t)
    diffusion!(dw, w, plate, prop, boundary)
end

using OrdinaryDiffEq
θinit = 400ones(N[1]*N[2]) 
tspan = (0.0, 2e5)
tsave = 1000.0;
alg = KenCarp5()
prob = ODEProblem(heat_conduction!,θinit,tspan)
sol = solve(prob,alg, saveat=tsave)

emit(sol[1,end], emission)

using Plots
plot(sol[1,:],legend=false)

plot(sol[1:end-1,end])
plot((sol[1:end-1,end] - sol[2:end,end]))

temp2d = reshape(sol[:,10],N...)
contourf(temp2d)