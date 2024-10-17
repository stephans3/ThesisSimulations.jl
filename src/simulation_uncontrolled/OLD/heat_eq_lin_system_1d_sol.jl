#=
    Θ'(t) = α/Δx^2 * D Θ(t)

    Solution is:
    Θ(t) = exp(α/Δx^2 * D*t)*Θ(0)

=#

θ₀ = 300.0  # Initial temperature
λ = 45.0    # Thermal conductivity: constant
ρ = 7800.0  # Mass density: constant
c = 480.0   # Specific heat capacity: constan
α = λ/(ρ*c) # Diffusivity

L = 0.2  # Length
Nx = 40  # Number of Nodes
Δx = L/Nx

Tf = 300;
t_samp = 1.0;
tspan = (0.0, Tf)
tgrid = 0 : t_samp : Tf;

# Initial temperature distribution + θ₀
p = 3e4 #
temp_init(x) = p*x*(L-x);

D = zeros(Int64,Nx, Nx)
for i=2:Nx-1
    D[i,i-1 : i+1] = [1,-2,1];
end

D[1+di,1+di:2+di] = [-1,1]
D[Nx+di,Nx-1+di:Nx+di] = [1,-1]

xgrid = L/(2Nx) : L/Nx : L
θinit = θ₀ .+ temp_init.(xgrid)

using LinearAlgebra

# Eigenvalues + Eigenvectors
evals, E2 = eigen(D) 
E1 = diagm(evals);
E2i = inv(E2)

# Analytical solution of ODE
temp_sol(t) = E2*exp(α/Δx^2 *E1*t)*E2i*θinit

temp_sol_data = temp_sol.(tgrid)
temp_true_data = hcat(temp_sol_data...)
using Plots
plot(hcat(temp_sol_data...)', legend=false)

Δtmin = Δx^2 / (2*α)

dt = t_samp # 1.1*Δtmin;
tsteps_euler = 0 : dt : Tf;

temp_euler = zeros(length(xgrid), length(tsteps_euler))
temp_euler[:,1] = θinit

for i=2 : length(tsteps_euler)
    temp_euler[:,i] = (I + dt*α/Δx^2*D)*temp_euler[:,i-1] 
end

plot(tsteps_euler[1:10:80],temp_euler[:,1:10:80]', legend=false)
plot(tsteps_euler[1:10:end],temp_euler[:,1:10:end]', legend=false)

err = temp_true_data - temp_euler
plot(tsteps_euler,err', legend=false)

using OrdinaryDiffEq
alg = KenCarp5()
heat_eq_ode(x,p,t) = α/Δx^2*D*x
prob = ODEProblem(heat_eq_ode,θinit,tspan)
sol = solve(prob,alg, saveat=t_samp)

data_kc = Array(sol)

err_kc = temp_true_data - data_kc
plot(tsteps_euler,err_kc[1:10:end,:]', legend=false)
