
using LinearAlgebra  

p = 1;

A = p*[-0 0 0;
     0 -1 0;
     0 0 -3];

psi(j,n) = cospi((j-1)*(2n-1)/6)

nv = 1:3
ψ₁ = psi.(1,nv) / norm(psi.(1,nv))
ψ₂ = psi.(2,nv) / norm(psi.(2,nv))
ψ₃ = psi.(3,nv) / norm(psi.(3,nv))

V_normed = hcat(ψ₁, ψ₂, ψ₃)

Δt = 10 # 2/3 # 0.5 # 2/3 # 1 # 2/3 #0.1 # 1/3

M1 = (I+Δt*A)
M2 = inv(I-Δt*A)
M3 = inv(I-Δt*A)*(I+Δt*A)

Nt = 10;
tsteps = 0:Δt:Nt*Δt;
zdata1 = zeros(3,length(tsteps)+1)
zdata2 = zeros(3,length(tsteps)+1)
zdata3 = zeros(3,length(tsteps)+1)

x0 = [1,2,-1]
zdata1[:,1] = V_normed*x0
zdata2[:,1] = V_normed*x0
zdata3[:,1] = V_normed*x0
     
# Euler algorithm
for (idx,t) in enumerate(tsteps)
    # zdata1[:, idx+1] = M1*zdata1[:,idx]
    zdata2[:, idx+1] = M2*zdata2[:,idx]
    zdata3[:, idx+1] = M3*zdata3[:,idx]
end


using Plots
plot(zdata1')
plot(zdata2')
plot(zdata3')