
using BellBruno
N_der = 10;              # Number of derivatives

# Either create new Bell polynomials  ones
bp = bell_poly(N_der);   # Create bell polynomials

# Or load already existing Bell polynomial numbers
# bp_load = BellBruno.read_bell_poly()
# bp = bp_load[1:N_der]

# Compute Bell coefficients
bc = bell_coeff(bp);     # Compute bell coefficients

#=
    f(z) = -1 z^(-2)
    d^n/dz^n f(z) = -1 * z^(-2-n) π(-2-j) for j=0:1:n
=#
function outer_fun!(y, z, p) 
    c = -1;

    y[1] = simple_monomial(z, c, p)

    fi = firstindex(y)
    li = lastindex(y)

    for idx in fi:li-1
        y[idx+1] = simple_monomial_der(z, c, p, idx)
    end
end

#=
    g(t)  = t/T - (t/T)^2
    g'(t) = 1/T - 2*t/T^2
    g''(t) = -2/T^2
    g^(n)(t) = 0 for n>2
=#
function inner_fun!(x, t :: Float64; T = 1.0 :: Real)
    c₁ = 0;
    c₂ = 1/T;
    c₃ = -1/T^2;

    x[1] = c₁ + c₂*t + c₃*t^2;  # g(t)  = t/T - (t/T)^2
    x[2] = c₂ + 2*c₃*t;         # g'(t) = 1/T - (2/T^2)*t
    x[3] = 2*c₃;                # g''(t)= -2/T^2
    x[4:end] .= 0;              # g^(n)(t) = 0 for n>2
end



function build_derivative(n_der, bc, bp, data_inner, data_outer, tgrid)
    nt = size(data_inner)[1]
    res = zeros(nt)

    data_out_is_vector = false;

    if length(data_outer[1,:]) == 1
        data_out_is_vector = true
    end

    for k=1 : n_der
        fi = firstindex(bp[n_der+1][k][1,:])           
        li = lastindex(bp[n_der+1][k][1,:])
        sol_prod = zeros(BigFloat,nt)   # Solution of the product π
        for μ = fi : li
                
            sol_prod_temp = zeros(BigFloat,nt)
                
            a = bc[n_der+1][k][μ]   # Coefficients
                
            for (idx, _) in enumerate(tgrid)
                @views x = data_inner[idx,:]
                sol_prod_temp[idx] = a * mapreduce(^, *, x, bp[n_der+1][k][:,μ])
            end
            sol_prod = sol_prod + sol_prod_temp
        end

        if data_out_is_vector == true
            res = res + data_outer.*sol_prod
        else
            res = res + data_outer[:,k+1].*sol_prod
        end
    end

    return res
end


function compute_derivatives(n_der, bc, bp, T, dt; w=2)
    tgrid = dt : dt : T-dt; # Time grid
    nt = length(tgrid)      # Number of time steps
    
    # Outer derivatives
    g̃ = zeros(nt, n_der+1); # g̃_n(t) := d^n/dt^n g(t)
    f̃ = zeros(nt, n_der+1); # f̃_n(t) := d^n/dy^n f(z)
    
    for (idx, elem) in enumerate(tgrid)
        @views x = g̃[idx,:]
        @views y = f̃[idx,:]
        inner_fun!(x, elem, T=T)
        outer_fun!(y, x[1], -w) 
    end
    
    q = zeros(nt, n_der);
    h = zeros(BigFloat,nt, n_der+1);
    h[:,1] = exp.(big.(f̃[:,1]))
     
    for n = 1 : n_der
        q[:,n] = build_derivative(n, bc, bp, g̃[:,2:end], f̃, tgrid)
        h[:,n+1] = build_derivative(n, bc, bp, q, h[:,1], tgrid)
        println("Iteration n= ", n)
    end

    return h
end

p_cntr = 2.0
Tf = 3000;    # Final simulation time
N_dt = 3000;
dt = Tf/N_dt; # Sampling time
h_data = compute_derivatives(N_der, bc, bp, Tf, dt, w=p_cntr);



using FastGaussQuadrature
bump(t) = exp(-1 / (t/Tf - (t/Tf)^2)^p_cntr)
t_gq, weights_gq = FastGaussQuadrature.gausslegendre(1000)
tshift = Tf/2;
ω_int = tshift *FastGaussQuadrature.dot( weights_gq ,bump.(tshift*t_gq .+ tshift))


function ref(t)
    ts1 = t/2
    if t <= 0
        # println("1")
        return 0
    elseif t >= Tf
        # println("2")
        return 1
    else
        # println("3")
        return ts1*FastGaussQuadrature.dot( weights_gq ,bump.(ts1*t_gq .+ ts1))/ω_int;
    end
end

tgrid1 = dt : dt : Tf-dt; # Time grid
ref_init = 300; # Intial Temperature
diff_ref = 100; # (y_f - y_0) = 100 Kelvin
ref_data = diff_ref*ref.(tgrid1)
ref_raw = hcat(ref_data,(diff_ref*h_data)/ω_int)

##################
#=
- Compute input signal u(t) for aluminum and steel 38Si7
=#

L = 0.1; # Length of 1D rod
# Aluminium
λ = 50;  # Thermal conductivity
ρ = 8000; # Density
c = 400;  # Specific heat capacity
α = λ / (ρ * c) # Diffusivity

Nx = 10;
Δx = L/Nx

D1 = zeros(Int64,Nx, Nx)
for i=2:Nx-1
    D1[i,i-1 : i+1] = [1,-2,1];
end
D1[1,1:2] = [-1,1]
D1[Nx,Nx-1:Nx] = [1,-1]

a1 = α/(Δx^2)
a1_inv = (Δx^2)/α 
A = α*D1/(Δx^2)
b1 = (Δx*c*ρ)
B = vcat(1, zeros(Int64,Nx-1)) / b1
C = hcat(zeros(Int64,1,Nx-1),1)

Om = mapreduce(i-> C*D1^i,vcat,0:Nx-1)
Om_inv = inv(Om)
Om_inv = (Om_inv' .* mapreduce(i-> a1_inv^i,vcat,0:Nx-1))'
Mu = hcat((a1*b1)*(-C*D1^Nx*Om_inv),b1*a1_inv^(Nx-1))
u_raw = (Mu*ref_raw[:,1:Nx+1]')'



function input_signal_fbc(t,u_data)
    if t <= 0
        return u_data[1]
    elseif t >= Tf
        return u_data[end]
    end
    dt = Tf/(length(u_data)-1)
    τ = t/dt + 1
    t0 = floor(Int, τ)
    t1 = t0 + 1;

    u0 = u_data[t0]
    u1 = u_data[t1]

    a = u1-u0;
    b = u0 - a*t0

    return a*τ + b;
end

function input_obc(t,p)
    return exp(p[1] - p[3]^2 * (t/Tf - 1/p[2])^2)
end


# u - Optimization values
# p - addtional/known parameters
function loss_optim_L1(u,p)
    input_oc(t) = input_obc(t,[p[1], p[2], u[1]])
    input_error = u_fbc-input_oc.(tgrid1)
    err = sum(abs, input_error)^2 / sum(abs, u_fbc)^2
    return err
end

function loss_optim_L2(u,p)
    input_oc(t) = input_obc(t,[p[1], p[2], u[1]])
    input_error = u_fbc-input_oc.(tgrid1)
    err = sum(abs2, input_error) / sum(abs2, u_fbc)
    return err
end

function loss_optim_Linf(u,p)
    input_oc(t) = input_obc(t,[p[1], p[2], u[1]])
    input_error = u_fbc-input_oc.(tgrid1)
    err = maximum(abs.(input_error))^2 / maximum(abs.(u_fbc))^2
    return err
end

tgrid1
u_fbc = Float64.(max.(u_raw,0))
p1 = Float64(log(maximum(u_fbc)))
t_max = tgrid1[argmax(u_fbc)]
p2 = Tf / t_max

p3grid = 1:0.01:20
opt_costs_L1 = mapreduce(p3 -> loss_optim_L1(p3,[p1,p2]),vcat, p3grid)
opt_costs_L1 = Float64.(opt_costs_L1)

opt_costs_L2 = mapreduce(p3 -> loss_optim_L2(p3,[p1,p2]),vcat, p3grid)
opt_costs_L2 = Float64.(opt_costs_L2)

opt_costs_Linf = mapreduce(p3 -> loss_optim_Linf(p3,[p1,p2]),vcat, p3grid)
opt_costs_Linf = Float64.(opt_costs_Linf)



using ForwardDiff
loss_L1(z) = loss_optim_L1(z,[p1,p2])
loss_L2(z) = loss_optim_L2(z,[p1,p2])
loss_Linf(z) = loss_optim_Linf(z,[p1,p2])

loss_grad_L1 = mapreduce(p3 -> ForwardDiff.gradient(loss_L1, [p3]),vcat,p3grid)
loss_grad_L2 = mapreduce(p3 -> ForwardDiff.gradient(loss_L2, [p3]),vcat,p3grid)
loss_grad_Linf = mapreduce(p3 -> ForwardDiff.gradient(loss_Linf, [p3]),vcat,p3grid)



# Find optimal p3
using Optimization, OptimizationOptimJL
opt_p = [p1, p2]
opt_u0 = [2.0]
loss_optim_L2(opt_u0, opt_p)

optf_L1 = OptimizationFunction(loss_optim_L1, Optimization.AutoForwardDiff())
opt_prob_L1 = OptimizationProblem(optf_L1, opt_u0, opt_p)
p3_opt_L1 = solve(opt_prob_L1, ConjugateGradient())[1]

optf_L2 = OptimizationFunction(loss_optim_L2, Optimization.AutoForwardDiff())
opt_prob_L2 = OptimizationProblem(optf_L2, opt_u0, opt_p)
p3_opt_L2 = solve(opt_prob_L2, ConjugateGradient())[1]

optf_Linf = OptimizationFunction(loss_optim_Linf, Optimization.AutoForwardDiff())
opt_prob_Linf = OptimizationProblem(optf_Linf, opt_u0, opt_p, lb=[1.0], ub=[Inf])
p3_opt_Linf = solve(opt_prob_Linf, ConjugateGradient())[1]

u_obc_L1 = mapreduce(t->input_obc(t,[p1,p2,p3_opt_L1]), vcat, tgrid1)
u_obc_L2 = mapreduce(t->input_obc(t,[p1,p2,p3_opt_L2]), vcat, tgrid1)
u_obc_Linf = mapreduce(t->input_obc(t,[p1,p2,p3_opt_Linf]), vcat, tgrid1)

err_L1 = u_fbc-u_obc_L1
err_L2 = u_fbc-u_obc_L2
err_Linf = u_fbc-u_obc_Linf

p3opt = [p3_opt_L1,p3_opt_L2,p3_opt_Linf]
loss_opt = [loss_L1(p3_opt_L1), loss_L2(p3_opt_L2), loss_Linf(p3_opt_Linf)]

using CairoMakie

path2folder = "results/figures/controlled/"
filename = path2folder*"optim_based_1d_costs.pdf"
begin   
    f = Figure(size=(1300,400),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel = L"Parameter $p_{3}$", ylabel = "Objective", 
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    ax1.xticks = 2 : 2 : 20;
    # ax1.yticks = 0 : 2.5 : 10;
    #ax1.yticks = -10 : 5 : 20;
    lines!(p3grid[250:end], opt_costs_L1[250:end];   linestyle = :dot,   linewidth = 5, label=L"$L_{1}$")
    lines!(p3grid, opt_costs_L2;   linestyle = :dash,  linewidth = 5, label=L"$L_{2}$")
    lines!(p3grid, opt_costs_Linf; linestyle = :solid, linewidth = 5, label=L"$L_{\infty}$")

    axislegend(; position = :rc, backgroundcolor = (:grey90, 0.1), labelsize=30);
    
    ax2 = Axis(f[1, 2], xlabel = L"Parameter $p_{3}$", ylabel = L"Objective $\times 10^{-3}$",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)
    #ax2.xticks = 2 : 2 : 20;

    yscale = 1e3;
    p3sub = 1100:1175
    ax2.xticks = 12 : 0.25 : 12.75;
    lines!(p3grid[p3sub], yscale*opt_costs_L1[p3sub];   linestyle = :dot,   linewidth = 5, label=L"$L_{1}$")
    lines!(p3grid[p3sub], yscale*opt_costs_L2[p3sub];   linestyle = :dash,  linewidth = 5, label=L"$L_{2}$")
    lines!(p3grid[p3sub], yscale*opt_costs_Linf[p3sub]; linestyle = :solid, linewidth = 5, label=L"$L_{\infty}$")
    scatter!(p3opt,yscale*loss_opt, marker = :xcross, markersize=25, color=:purple)
    axislegend(; position = :rb, backgroundcolor = (:grey90, 0.1), labelsize=30);

    f
    save(filename, f, pt_per_unit = 1)       
end


filename = path2folder*"optim_based_1d_costs_gradient.pdf"
begin   
    f = Figure(size=(1300,400),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel = L"Parameter $p_{3}$", ylabel = "Gradient",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    ax1.xticks = 2 : 2 : 20;
    # ax1.yticks = 0 : 2.5 : 10;

    lines!(p3grid[250:end], loss_grad_L1[250:end];   linestyle = :dot,   linewidth = 5, label=L"$L_{1}$")
    lines!(p3grid, loss_grad_L2;   linestyle = :dash,  linewidth = 5, label=L"$L_{2}$")
    lines!(p3grid, loss_grad_Linf; linestyle = :solid, linewidth = 5, label=L"$L_{\infty}$")

    axislegend(; position = :rb, backgroundcolor = (:grey90, 0.1), labelsize=30);
    
    ax2 = Axis(f[1, 2], xlabel = L"Parameter $p_{3}$", ylabel = L"Gradient $\times 10^{-3}$",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    yscale = 1e3;
    p3sub = 1100:1175
    ax2.xticks = 12 : 0.25 : 12.75;
    lines!(p3grid[p3sub], yscale*loss_grad_L1[p3sub];   linestyle = :dot,   linewidth = 5, label=L"$L_{1}$")
    lines!(p3grid[p3sub], yscale*loss_grad_L2[p3sub];   linestyle = :dash,  linewidth = 5, label=L"$L_{2}$")
    lines!(p3grid[p3sub], yscale*loss_grad_Linf[p3sub]; linestyle = :solid, linewidth = 5, label=L"$L_{\infty}$")
    scatter!(p3opt,zeros(3), marker = :xcross, markersize=25, color=:purple)
    axislegend(; position = :rb, backgroundcolor = (:grey90, 0.1), labelsize=30);

    f  
    save(filename, f, pt_per_unit = 1)   
end



filename = path2folder*"optim_based_1d_input.pdf"
begin   
    f = Figure(size=(600,400),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel = "Time t in [s]", ylabel=L"Input $\times 10^4$",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    scale = 1e-4;

    ax1.xticks = 0 : 500 : 3000;
    ax1.yticks = 0 : 2.5 : 10;

    tstart = 450;
    tend = 2550;
    lines!(tgrid1[tstart:tend], scale*u_fbc[tstart:tend,1];   linestyle = :solid, color=Makie.wong_colors()[4] ,   linewidth = 3, label="FBC")

    samp1 = tstart:3*43:tend
    samp2 = tstart:3*51:tend
    samp3 = tstart:3*59:tend

    scatter!(tgrid1[samp1], scale*u_obc_L1[samp1,1]; markersize=15, color=Makie.wong_colors()[1] ,  label=L"$L_{1}$")
    scatter!(tgrid1[samp2], scale*u_obc_L2[samp2,1]; markersize=15, color=Makie.wong_colors()[2], marker=:rect ,label=L"$L_{2}$")
    scatter!(tgrid1[samp3], scale*u_obc_Linf[samp3,1]; markersize=15, color=Makie.wong_colors()[3], marker=:cross ,label=L"$L_{\infty}$")
    
    axislegend(; position = :rc, backgroundcolor = (:grey90, 0.1), labelsize=30);
      
    f  
    # save(filename, f, pt_per_unit = 1)   
end



filename = path2folder*"optim_based_1d_error.pdf"
begin   
    f = Figure(size=(600,400),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel = "Time t in [s]", ylabel=L"Error $\times 10^3$",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    scale = 1e-3;

    ax1.xticks = 500 : 500 : 3000;
    #ax1.xticks = 0 : 750 : 3000;
    #ax1.yticks = 0 : 2.5 : 10;
    tstart = 450;
    tend = 2550;
    lines!(tgrid1[tstart:tend], scale*err_L1[tstart:tend];   linestyle = :dot,   linewidth = 5, label=L"$L_{1}$")
    lines!(tgrid1[tstart:tend], scale*err_L2[tstart:tend];   linestyle = :dash,   linewidth = 5, label=L"$L_{2}$")
    lines!(tgrid1[tstart:tend], scale*err_Linf[tstart:tend]; linestyle = :solid,   linewidth = 5, label=L"$L_{\infty}$")
  
    axislegend(; position = :rb, backgroundcolor = (:grey90, 0.1), labelsize=30);
      
    f  
    # save(filename, f, pt_per_unit = 1)   
end







