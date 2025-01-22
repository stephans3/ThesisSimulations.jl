
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


λ = 1;
ρ = 1; # Density
c = 1;  # Specific heat capacity

L = 1
N = 100;
Δx = L/N


α = λ/(ρ*c)

using LinearAlgebra
p1 = α/(Δx^2) 
μ = eigvals_he(N, 1, 1, [p1, 0,0])
V = eigvecs_he(N, 1, 1)

Tf = 2.0

function M(t)
    vec = zeros(N)
    vec[1] = t
    for n=2:N
        vec[n] = (1-exp(μ[n]*t))/abs(μ[n])
    end
    return vec
end



E = vcat(1, zeros(N-1))

θ0 = zeros(N);
u_in = 1;
eN = vcat(zeros(N-1),1)

function heat_solution(t)
    return V *exp(diagm(μ)*t) * V' * θ0 +  V*diagm(M(t))*V'*E*u_in/Δx
end

function y_output(t)
    return eN'*V *exp(diagm(μ)*t) * V' * θ0 +  eN'*V*diagm(M(t))*V'*E*u_in/Δx
end

function yout_der(t)
    return eN'*V*diagm(μ)*exp(diagm(μ)*t) * V' * θ0 +  eN'*V*exp(diagm(μ)*t)*V'*E*u_in/Δx
end

tgrid = 0:0.01:1.3

y_out_data = y_output.(tgrid)
y_der_data = yout_der.(tgrid)


using CairoMakie
path2folder = "results/figures/controlled/feedback/"
begin
    # filename = path2folder*"step_response.pdf"
    filename = path2folder*"step_response_latex.pdf"

    f = Figure(size=(600,400),fontsize=26)
    # ax1 = Axis(f[1, 1], xlabel = "Time in [s]", ylabel = "", xlabelsize = 30, ylabelsize = 30, xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1)
    
    ax1 = Axis(f[1, 1], xlabel = L"Time in [s] $$", ylabel = "", xlabelsize = 30, ylabelsize = 30, xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1)
    
    ax1.xticks = collect(0:0.2:1.2)
    lines!(tgrid, y_out_data, linestyle = :solid,  linewidth = 5, label=L"Output $y$")
    lines!(tgrid, y_der_data, linestyle = :dash,  linewidth = 5, label=L"Derivative $\frac{d}{dt}y$")

    axislegend(; position = :rb, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f

    save(filename, f, pt_per_unit = 1)   
end


