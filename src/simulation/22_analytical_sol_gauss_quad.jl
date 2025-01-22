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

Tf = 10.0 # 10.0; # 20.0; # Final time

using FastGaussQuadrature
s,w = FastGaussQuadrature.gausslegendre(1000)
p1 = Tf/2

ϕ(t) = 1.2exp(-(0.7*(t-(Tf/3)))^2) #-0.01*(t-Tf/2)^2 + 1 #  exp(-((t-(Tf/2))/4)^2)

gen(t) = [1, exp(-t), exp(-3t)]

E = [1, 0, 0]
VE = V'*E

int_kernel(τ,i) = gen(-τ)[i]*(VE*ϕ(τ))[i]

t₁=0.1
t₂=0.2 
x₁=[1,2,3]
ΔT = (t₂-t₁)/2
τ = ΔT*s .+ (t₂+t₁)/2


function trajectory(t₁, t₂, x₁)
    ΔT = (t₂-t₁)/2
    τ = ΔT*s .+ (t₂+t₁)/2;
    integ1 = ΔT*w'*gen(t₂)[1]*int_kernel.(τ,1)
    integ2 = ΔT*w'*gen(t₂)[2]*int_kernel.(τ,2)
    integ3 = ΔT*w'*gen(t₂)[3]*int_kernel.(τ,3)
    Iv = [integ1,integ2,integ3]

    #sol = V * diagm(gen(t)) *V' * x0 + V*Iv
    sol = diagm(gen(t₂-t₁)) * x₁ + Iv
    return sol
end

tgrid = 0 : 0.1 : Tf
x0 = zeros(3) # [1.0,2,-1]
x₁ = V'*x0

sol_data = zeros(length(tgrid),length(x₁))
sol_data[1,:] = x₁
t0 = 0

for (idx, t) in enumerate(tgrid[2:end])
   
    sol_traj = trajectory(t0, t, x₁)
    sol_data[idx+1,:] = sol_traj 
    x₁ = sol_traj
    t0 = t
end

sol_orig = (V * sol_data')'


using CairoMakie
base_path = "results/figures/simulation/"

begin
    fig1 = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(fig1[1, 1], xlabel = "Time t in [s]", ylabel = "", 
        xlabelsize = 30, ylabelsize = 30, xgridstyle = :dash, ygridstyle = :dash, 
        xtickalign = 1., xticksize = 10, 
        xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,
        yminorgridvisible = true, yminorticksvisible = true, yminortickalign = 1,
        ytickalign = 1, yticksize = 10, xlabelpadding = 0,limits = (nothing, (-0.1, 2)))
    
    ax1.xticks = 0 : 1 : Tf;
    ax1.yticks = 0 : 0.5 : 2;
  
    lines!(ax1, tgrid, sol_data[:,1];linestyle = :dot,   linewidth = 5, label = L"\tilde{\Theta}_{1}")
    lines!(ax1, tgrid, sol_data[:,2];linestyle = :dash,  linewidth = 5, label = L"\tilde{\Theta}_{2}")
    lines!(ax1, tgrid, sol_data[:,3];linestyle = :solid, linewidth = 5, label = L"\tilde{\Theta}_{3}")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);
    fig1
    save(base_path*"linear_he_example_sol_transformed.pdf", fig1, pt_per_unit = 1)    
end


begin
    fig1 = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(fig1[1, 1], xlabel = "Time t in [s]", ylabel = "", 
       xlabelsize = 30, ylabelsize = 30, xgridstyle = :dash, ygridstyle = :dash, 
       xtickalign = 1., xticksize = 10, 
       xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,
       yminorgridvisible = true, yminorticksvisible = true, yminortickalign = 1,
       ytickalign = 1, yticksize = 10, xlabelpadding = 0,limits = (nothing, (-0.1, 1.5)))
   
       ax1.xticks = 0 : 1 : Tf;
       ax1.yticks = 0 : 0.5 : 2;
 
    lines!(ax1, tgrid, sol_orig[:,1];linestyle = :dot,   linewidth = 5, label = L"\Theta_{1}")
    lines!(ax1, tgrid, sol_orig[:,2];linestyle = :dash,  linewidth = 5, label = L"\Theta_{2}")
    lines!(ax1, tgrid, sol_orig[:,3];linestyle = :solid, linewidth = 5, label = L"\Theta_{3}")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);
    fig1
    save(base_path*"linear_he_example_sol_original.pdf", fig1, pt_per_unit = 1)    
end

#=
begin
    fig1 = Figure(size=(400,300),fontsize=26)
    ax1 = Axis(fig1[1, 1], ylabelsize = 22, xlabel = "Time t in [s]", ylabel = "", 
       xlabelsize = 24, xgridstyle = :dash, ygridstyle = :dash, 
       xtickalign = 1., xticksize = 10, 
       xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,
       yminorgridvisible = true, yminorticksvisible = true, yminortickalign = 1,
       ytickalign = 1, yticksize = 10, xlabelpadding = 0)
   
       ax1.xticks = 0 : 1 : Tf;
       ax1.yticks = 0 : 0.2 : 2;
 
    lines!(ax1, tgrid, ϕ.(tgrid);linestyle = :dash,   linewidth = 5, label = L"\phi_{1,1}")
    # lines!(ax1, tgrid, sol_orig[:,2];linestyle = :dash,  linewidth = 5, label = L"\Theta_{2}")
    # lines!(ax1, tgrid, sol_orig[:,3];linestyle = :solid, linewidth = 5, label = L"\Theta_{3}")
    axislegend(; position = :rt, backgroundcolor = (:grey90, 0.1));
    fig1
    save(base_path*"linear_he_example_phi_flux.pdf", fig1, pt_per_unit = 1)    
end
=#
