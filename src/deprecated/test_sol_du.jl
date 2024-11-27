function lorenz(du,u,p,t)
    nonlinear(du,u)
end

function nonlinear(du,u)
    du[1] = 10.0*(u[2]-u[1])
    du[2] = u[1]*(28.0-u[3]) - u[2]
    du[3] = u[1]*u[2] - (8/3)*u[3]
end

u0 = [1.0;0.0;0.0]
tspan = (0.0,100.0)
prob = ODEProblem(lorenz,u0,tspan)
alg = KenCarp5()
ts = tspan[1] : 2.0 : tspan[end]
sol = solve(prob,alg, saveat=ts)

sol.k

# [sol.k[i][1] for i in 1:length(sol)]

