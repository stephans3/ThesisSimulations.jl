

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

Cd = diagm(ones(3))
C = hcat(zeros(Int64,N₁,N₁*(N₂-1)),Cd);

# Original

Om = mapreduce(i-> C*A^i,vcat,0:N₂-1)
Om_inv = inv(Om)

Mu1 = inv(C*A^(N₂-1)*B)
Mu = hcat(-Mu1*C*A^N₂ * Om_inv, Mu1)
sum(Mu, dims=1)


# Hyperbolic Tangent
f(t,p) = tanh(p*t)
d1_f(t,p) = p*(1-f(t,p)^2)
d2_f(t,p) = (p^2)*(-2f(t,p) + 2f(t,p)^3)
d3_f(t,p) = (p^3)*(-2 + 8f(t,p)^2 - 6f(t,p)^4)
d4_f(t,p) = (p^4)*(16f(t,p) - 40f(t,p)^3 + 24f(t,p)^5)
d5_f(t,p) = (p^5)*(16 - 136f(t,p)^2 + 240f(t,p)^4-120f(t,p)^6)


# Transition
ψ(t,T,p) = (f(t/T-0.5,p)+1)/2
d1_ψ(t,T,p) = d1_f(t/T-0.5,p) / (2*big(T))
d2_ψ(t,T,p) = d2_f(t/T-0.5,p) / (2*big(T)^2)
d3_ψ(t,T,p) = d3_f(t/T-0.5,p) / (2*big(T)^3)
d4_ψ(t,T,p) = d4_f(t/T-0.5,p) / (2*big(T)^4)
d5_ψ(t,T,p) = d5_f(t/T-0.5,p) / (2*big(T)^5)


Tf = 1200;
ps = 10; # steepness
tgrid = 0 : Tf/1000 : Tf

dψ1(t) = d1_ψ(t,Tf,ps)
dψ2(t) = d2_ψ(t,Tf,ps)
dψ3(t) = d3_ψ(t,Tf,ps)
dψ4(t) = d4_ψ(t,Tf,ps)
dψ5(t) = d5_ψ(t,Tf,ps)

θinit = 300;
Δr = 100;
ref(t) = θinit + Δr*ψ(t,Tf,ps)

r0 = repeat(ref.(tgrid),1,3)
r1 = Δr*repeat(dψ1.(tgrid),1,3)
r2 = Δr*repeat(dψ2.(tgrid),1,3)
r3 = Δr*repeat(dψ3.(tgrid),1,3)
r4 = Δr*repeat(dψ4.(tgrid),1,3)
r5 = Δr*repeat(dψ5.(tgrid),1,3)

ref_mimo = hcat(r0, r1, r2, r3, r4, r5)
u_raw = (Mu*ref_mimo')'

using Plots
plot(tgrid, u_raw)