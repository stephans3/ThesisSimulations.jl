# dt θ(t) = α/dx^2 D θ(t)

θ₀ = 300.0  # Initial temperature
λ = 45.0    # Thermal conductivity: constant
ρ = 7800.0  # Mass density: constant
c = 480.0   # Specific heat capacity: constan
α = λ/(ρ*c) # Diffusivity

L = 0.2  # Length
Nx = 40  # Number of Nodes
Δx = L/Nx


D = zeros(Int64,Nx, Nx)
j=1
di = (j-1)*Nx
for i=2:Nx-1
    D[i+di,i-1+di : i+1+di] = [1,-2,1];
end

D[1+di,1+di:2+di] = 2*[-1,1]
D[Nx+di,Nx-1+di:Nx+di] = 2*[1,-1]
 
# D[1+di,1+di:2+di] = [-2,1]
# D[Nx+di,Nx-1+di:Nx+di] = [1,-2]

#D = α/Δx^2 * D

using LinearAlgebra
# Initial temperature distribution + θ₀
p = 3e4 #
temp_init(x) = p*x*(L-x);
xgrid = L/(2Nx) : L/Nx : L
θinit = θ₀ .+ temp_init.(xgrid)
evals, E2 = eigen(D) 
E1 = diagm(evals);
E2i = inv(E2)
# Analytical solution of ODE
temp_sol1(t) = E2*exp(α/Δx^2 *E1*t)*E2i*θinit

(Es1, Es2, es3) = schur(D)
Es2i = inv(Es2)
temp_sol2(t) = Es2*exp(α/Δx^2 *Es1*t)*Es2i*θinit

round.(exp(Es1*50), digits=10)

exp([2 1 1;0 2 1; 0 0 2])

using Plots
err_schur = temp_sol1(1) - temp_sol2(1)
plot(err_schur)
plot(Es2[:,end-1])



ev_D = eigvals(D)

ev1 = zeros(Nx)
for i=1:Nx
ev1[i] = -2-2*sqrt(1*1)*cos(i*pi/(Nx+1))
end

using Plots
err = ev_D - ev1
scatter(err)

schur(D)

D- D'

DT = diagm(ones(Nx));
b1 = vcat(2, ones(Int64,Nx-2))
c1 = vcat(ones(Int64,Nx-2),2)

for i=2:Nx
    DT[i,i] = sqrt(reduce(*,c1[1:i-1])/reduce(*,b1[1:i-1]))
end

J = inv(DT)*D*DT
exp(J) - exp(D)

evecs_D = eigvecs(D)


evec_max_arr[200]
plot(evec_max_arr)
plot!( pi/Nx_arr[end] .+ sqrt.(pi ./ (2Nx_arr)))

err = pi/Nx_arr[end] .+ sqrt.(pi ./ (2Nx_arr)) - evec_max_arr
plot(err)
plot(evec_max_arr[2:end]-evec_max_arr[1:end-1])



ev1 = zeros(Nx)
for i=1:Nx
ev1[i] = -2-2*sqrt(1*1)*cos(i*pi/(Nx+1))
end

dummy = -sinpi.(2*(1:Nx)./(Nx+1)) * (2pi/(Nx+1))
plot(dummy)


