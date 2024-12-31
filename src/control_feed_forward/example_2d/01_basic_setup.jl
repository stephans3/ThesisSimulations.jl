# Compute thermal conductivity
# x₁ direction

θvec = collect(300:50:500)
f(z) = [1  z  z^2 z^3 z^4]
M_temp = mapreduce(z-> f(z), vcat, θvec)

λ1data = [40,44,50,52,52.5]
λ1p = inv(M_temp)*λ1data

using LinearAlgebra
λ1p11 = diagm([1,1e2,1e4,1e6,1e8])*λ1p
λ1p11 = round.(λ1p11, digits=3)



θgrid = 300 : 10 : 500;
λ1graph =  vcat(f.(θgrid)...)*λ1p

using Plots
plot(θgrid,λ1graph)


λ2data = [40,55,60,65,68]
λ2p = inv(M_temp)*λ2data
λ2graph =  vcat(f.(θgrid)...)*λ2p

λ2p22 = diagm([1,1e2,1e4,1e6,1e8])*λ2p
λ2p22 = round.(λ2p22, digits=3)

plot(θgrid,λ2graph)
