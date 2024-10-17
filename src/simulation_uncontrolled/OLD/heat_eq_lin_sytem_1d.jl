# dt θ(t) = α/dx^2 D θ(t)

θ₀ = 300.0  # Initial temperature
λ = 45.0    # Thermal conductivity: constant
ρ = 7800.0  # Mass density: constant
c = 480.0   # Specific heat capacity: constan
α = λ/(ρ*c) # Diffusivity

L = 0.2  # Length
Nx = 200  # Number of Nodes
Δx = L/Nx


Nx_arr = collect(3:1:300)
evec_max_arr = zeros(length(Nx_arr));

for (n_nx, Nx) in enumerate(Nx_arr)

D = zeros(Int64,Nx, Nx)
j=1
di = (j-1)*Nx
for i=2:Nx-1
    D[i+di,i-1+di : i+1+di] = [1,-2,1];
end

D[1+di,1+di:2+di] = [-1,1]
D[Nx+di,Nx-1+di:Nx+di] = [1,-1]

# D[1+di,1+di:2+di] = [-2,1]
# D[Nx+di,Nx-1+di:Nx+di] = [1,-2]

D = α/Δx^2 * D

using LinearAlgebra
ev_D = eigvals(D)
evecs_D = eigvecs(D)

evec_max_arr[n_nx] = maximum(evecs_D[:,1])
end

plot(evecs_D[:,1])


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

using Plots
err = ev_D - ev1


scatter(err)

