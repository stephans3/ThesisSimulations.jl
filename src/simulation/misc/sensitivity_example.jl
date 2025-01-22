A = [-1 2;
    0 -3]

using LinearAlgebra
eigvals(A)

z0 = [1,1]
p0 = 2;
ana_sol(t,z0,p) = exp(p*A*t)*z0
sens_sol(t,z0,p) = A*t*ana_sol(t,z0,p)

ana_sol(1,z0,p0)
sens_sol(0.1,z0,p0)

using Plots
tgrid = 0 : 0.01 : 3;
sens1(t) = sens_sol(t,z0,p0)
erg_sens = sens1.(tgrid)

sens_data = vcat(erg_sens'...)

plot(tgrid, sens_data)

