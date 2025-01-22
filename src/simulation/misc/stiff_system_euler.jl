
p=1;

A = p* [-0 0 0;
     0 -1 0;
     0 0 -3]

A1 = p* [-1 1 0;
     1 -2 1;
     0 1 -1]


#=
x'(t) = A x(t)
V x'(t)  = A V x(t)

z = V x
z'(t) = V x'(t) = V A x(t)
z'(t) = V A V⁻¹ z(t)
=#

psi(j,n) = cospi((j-1)*(2n-1)/6)

V = [psi(1,1) psi(2,1) psi(3,1);
    psi(1,2) psi(2,2) psi(3,2);
    psi(1,3) psi(2,3) psi(3,3);]'

V'*V
V*V'


# V*A1*inv(V)

using LinearAlgebra
(I+0.01*A)[3,3]
(I+0.1*A)[3,3]
(I+0.2*A)[3,3]
(I+0.4*A)[3,3]


using LinearAlgebra
Δt = 1/3 # 0.5 # 2/3 # 1 # 2/3 #0.1 # 1/3
(I+Δt*A)

(I+Δt*A)^10
(I+Δt*A1)^10

#abs.(diag(I+Δt*A))



tsteps = 0:Δt:2;

xdata = zeros(3,length(tsteps)+1)
zdata = zeros(3,length(tsteps)+1)

x0 = [1,2,-1]
xdata[:,1] = x0
zdata[:,1] = V*x0

# Euler algorithm
for (idx,t) in enumerate(tsteps)
    xdata[:, idx+1] = (I+Δt*A1)*xdata[:,idx]
    zdata[:, idx+1] = (I+Δt*A)*zdata[:,idx]
end
zdata[:,1]
(I+Δt*A)
(I+Δt*A)*zdata[:,1]
xdata
zdata

using Plots
scatter(zdata')

exp(diagm(eigvals(A1)))

exp(A)

V1 = eigvecs(A1)

V*exp(A)*inv(V)
inv(V1)*exp(A)*V1
exp(A1)

V * xdata
inv(V) * zdata