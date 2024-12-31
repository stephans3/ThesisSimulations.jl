function alpha_coeff(θ)
    p = [370, -2.85, 8.458e-3, -1e-5, 4.1667e-9]
    N = length(p)
    
    ρ = 8000; # density
    c = 400; # specific heat capacity
    den = ρ*c;

    λ = mapreduce(i-> p[i]*θ^(i-1),+, 1:N)     
    α = λ/den
    return α
end

function beta_coeff(θ)
    p = [370, -2.85, 8.458e-3, -1e-5, 4.1667e-9]
    N = length(p)
    
    ρ = 8000; # density
    c = 400; # specific heat capacity
    den = ρ*c;

    dλ = mapreduce(i-> (i-1)*p[i]*θ^(i-2),+, 2:N)
    β = dλ / den

    return β
end


function semilin_he!(dw, w, p, t)
    N = length(w)

    α = alpha_coeff.(w)
    β = beta_coeff.(w)

    dw[1]   = 2α[1]*(-w[1]+w[2])/dx^2
    dw[N] = 2α[N]*( w[N-1]-w[N])/dx^2

    for i=2:N-1
        dw[i] = α[i]*(w[i-1] - 2w[i] + w[i+1])/dx^2 + β[i]*(-w[i-1] + w[i+1])^2 / (4*dx^2)
    end

    return nothing
end

L = 0.2; # length
Nx = 41;
dx = L/(Nx-1);

using OrdinaryDiffEq
θinit = 300ones(Nx)
θinit[1:20] .= 700
θinit[21] = 500

tspan = (0.0, 300)
tsave = 1.0;
alg = KenCarp5()
prob_dynamic = ODEProblem(semilin_he!,θinit,tspan)
sol_dynamic = solve(prob_dynamic, alg, saveat=tsave)

sol_dynamic[1]

using Plots
plot(sol_dynamic, legend=false)

plot(sol_dynamic[1])
for i=2:4:100
    display(plot!(sol_dynamic[i]))
end