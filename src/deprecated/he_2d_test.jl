
L = 0.3; # Length of 1D rod
W = 0.05
N₁ = 3;
N₂ = 5;
Nc = N₁*N₂ 
Δx₁ = L/N₁
Δx₂ = W/N₂

# Aluminium
λ₁ = 50;  # Thermal conductivity
λ₂ = 60;
ρ = 8000; # Density
c = 400;  # Specific heat capacity
α₁ = λ₁ / (ρ * c) # Diffusivity
α₂ = λ₂ / (ρ * c) # Diffusivity


# Diffusion matrices
D1 = zeros(Int64,Nc,Nc)
D2 = zeros(Int64,Nc,Nc)
for j=1:N₂
    di = (j-1)*N₁
    for i=2:N₁-1
        D1[i+di,i-1+di : i+1+di] = [1,-2,1];
    end
    D1[1+di,1+di:2+di] = [-1,1]
    D1[N₁+di,N₁-1+di:N₁+di] = [1,-1]
end


for i=1:N₁
    for j=2:N₂-1
        di = (j-1)*N₁
        D2[i+di,i+di-N₁:N₁:(i+di)+N₁] =  [1,-2,1]
        #D2[i+(j-1)*N₁,i-1+(j-1)*N₁ : i+1+(j-1)*N₁] = [1,-2,1];
    end
    D2[i,i:N₁:i+N₁] = [-1,1]
    D2[i+(N₂-1)*N₁,i+(N₂-2)*N₁:N₁:i+(N₂-1)*N₁] = [1,-1]
end

a1 = α₁/(Δx₁^2);
a2 = α₂/(Δx₂^2);


using LinearAlgebra
A = a1*D1 + a2*D2
b1 = (Δx₂*c*ρ)
# B = vcat(diagm(ones(Int64,N₁)), zeros(Int64,N₁*(N₂-1),N₁)) / b1

Bd = diagm(ones(3))/ b1 
B = vcat(Bd, zeros(Int64,N₁*(N₂-1),N₁)) 


function input_obc(t,p)
    return exp(p[1] - p[3]^2 * (t/Tf - 1/p[2])^2)
end


function heat_conduction!(dθ, θ, param, t)

    u1 = input_obc(t,param[1:3])
    u2 = input_obc(t,param[4:6])
    u3 = input_obc(t,param[7:9])
    
    u_in = [u1, u2, u3]
    dθ .= A*θ + B*u_in # B[:,1]*1 + B[:,2]*2 + B[:,3]*3 #*u_in
    # diffusion!(dθ, θ, plate, property, boundary, actuation, u_in)
end


Tf = 1200;
tspan = (0.0, Tf)
θ₀ = 300;
Ntotal = N₁*N₂
θinit = θ₀*ones(Ntotal)

dΘ = similar(θinit)

p_found = [11.81522609265776
            2.070393374741201
            9.212088025161702]
pinit = repeat(p_found,3)

heat_conduction!(dΘ,θinit,pinit,0)

# Δt = 1e-2               # Sampling time

using OrdinaryDiffEq

t_samp = 10.0
alg = KenCarp5()
prob = ODEProblem(heat_conduction!,θinit,tspan,pinit)
sol = solve(prob,alg,p=pinit, saveat=t_samp)