using Hestia
prop = StaticIsotropic(50, 8000, 400)

L = 0.1
N = 100
rod = HeatRod(L, N)

boundary = Boundary(rod)
h = 20;
Θamb = 300;
ε = 0.1;
em_total = Emission(h, Θamb,ε) 
# setEmission!(boundary, em_total, :west);
# setEmission!(boundary, em_total, :east);

function heat_conduction!(dw, w, param, t) 
    diffusion!(dw, w, rod, prop, boundary)
end


using OrdinaryDiffEq
xgrid = L/(2N) : L/N : L # Position in x-direction
θinit = 100*sinpi.(xgrid / L) # 600* ones(N) # Intial values
Tf    = 300 # 3000;
tspan =  (0.0, Tf)   # Time span
tsamp = 1.0;
alg = KenCarp4()    # Numerical integrator
prob_orig = ODEProblem(heat_conduction!,θinit,tspan)
sol_orig = solve(prob_orig, alg, saveat = tsamp)

using Plots
plot(sol_orig)

Nx=N
D1 = zeros(Int64,Nx, Nx)
for i=2:Nx-1
    D1[i,i-1 : i+1] = [1,-2,1];
end
D1[1,1:2] = [-1,1]
D1[Nx,Nx-1:Nx] = [1,-1]

α = 50/(8000*400)
Δx = L/N
a1 = α/(Δx^2)
a1_inv = (Δx^2)/α 
A = α*D1/(Δx^2)

aa_erg = A*Array(sol_orig)
plot(Array(sol_orig)[:,1])
plot(xgrid, aa_erg[:,10])


aa_erg1 = D1*Array(sol_orig)
plot(xgrid, aa_erg1[:,1])

maximum(sum(A*Array(sol_orig),dims=1))