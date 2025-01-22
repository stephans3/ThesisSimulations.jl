using LinearAlgebra

function eigvals_he(J, M, K, p)

    i(j,m,k) = j + (m-1)*J + (k-1)*M*J

    Nc = J*M*K
    μ = zeros(Nc)


    for j=1:J, m=1:M,k=1:K
        idx = i(j,m,k);
        μ[idx] = -2p[1]*(1-cospi((j-1)/J)) - 2p[2]*(1-cospi((m-1)/M)) - 2p[3]*(1-cospi((1-k)/K))
    end
    return μ
end

function eigvecs_he(J, M, K;orthonormal = true)

    f(z,n) = cospi((2n-1)*z)

    i(j,m,k) = j + (m-1)*J + (k-1)*M*J
    Nc = J*M*K
    V = zeros(Nc, Nc)

    for j=1:J, m=1:M,k=1:K
        idx = i(j,m,k);
        for n_j=1:J, n_m=1:M, n_k=1:K
            n_idx = i(n_j,n_m,n_k);
           
            ψ = f((j-1)/(2J), n_j)*f((m-1)/(2M),n_m)*f((k-1)/(2K),n_k)
            V[n_idx,idx] = ψ
        end

        if orthonormal == true
            V[:,idx] = V[:,idx] / norm(V[:,idx])
        end
    end
    return V
end

μ = eigvals_he(3, 1, 1, ones(3))
V = eigvecs_he(3, 1, 1)

Tf = 10.0; # 20.0; # Final time

using FastGaussQuadrature
s,w = FastGaussQuadrature.gausslegendre(1000)
p1 = Tf/2

ϕ(t) = exp(-(1*(t-(Tf/2)))^2) #-0.01*(t-Tf/2)^2 + 1 #  exp(-((t-(Tf/2))/4)^2)

gen(t) = [1, exp(-t), exp(-3t)]

E = [1, 0, 0]
VE = V'*E

int_kernel(τ,i) = gen(-τ)[i]*VE[i]*ϕ(τ)

#=
t = 3
p2 =  t/2
τ = p2*s .+ p2;

gen(t)[1]*int_kernel.(-τ,1)
gen(t)[2]*int_kernel.(-τ,2)
gen(t)[3]*int_kernel.(-τ,3)
=#

function trajectory(t, x0)
    p2 =  t/2
    τ = p2*s .+ p2;
    integ1 = p2*w'*gen(t)[1]*int_kernel.(τ,1)
    integ2 = p2*w'*gen(t)[2]*int_kernel.(τ,2)
    integ3 = p2*w'*gen(t)[3]*int_kernel.(τ,3)
    Iv = [integ1,integ2,integ3]

    #sol = V * diagm(gen(t)) *V' * x0 + V*Iv
    sol = diagm(gen(t)) * x0 + Iv
    return sol
end

tgrid = 0 : 0.1 : Tf
x0 = [1.0,2,-1]
my_sol(t) = trajectory(t, x0)
sol_data = my_sol.(tgrid)

sol_data1 = vcat(sol_data'...)

using Plots
plot(tgrid,sol_data1)

A = [-1 1 0;1 -2 1;0 1 -1]

function he_ode!(dθ, θ, p, t)
    dθ .= A*θ + E*ϕ(t)
end

using OrdinaryDiffEq
tspan = (0, Tf)
alg = Tsit5()
prob = ODEProblem(he_ode!,x0, tspan)
sol = solve(prob, alg,saveat=tgrid)

plot(sol)


err = vcat((sol_data - sol[:])'...)
plot(tgrid, err)

