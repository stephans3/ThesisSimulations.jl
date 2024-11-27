
L = 0.05; # Length of 1D rod
# Aluminium
λ = 60;  # Thermal conductivity
ρ = 8000; # Density
c = 400;  # Specific heat capacity
α = λ / (ρ * c) # Diffusivity

Nx = 5;
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





# Hyperbolic Tangent
f(t,p) = tanh(p*t)
d1_f(t,p) = p*(1-f(t,p)^2)
d2_f(t,p) = (p^2)*(-2f(t,p) + 2f(t,p)^3)
d3_f(t,p) = (p^3)*(-2 + 8f(t,p)^2 - 6f(t,p)^4)
d4_f(t,p) = (p^4)*(16f(t,p) - 40f(t,p)^3 + 24f(t,p)^5)
d5_f(t,p) = (p^5)*(16 - 136f(t,p)^2 + 240f(t,p)^4-120f(t,p)^6)


# Transition
ψ(t,T,p) = (f(t/T-0.5,p)+1)/2
d1_ψ(t,T,p) = d1_f(t/T-0.5,p) / (2*big(T))
d2_ψ(t,T,p) = d2_f(t/T-0.5,p) / (2*big(T)^2)
d3_ψ(t,T,p) = d3_f(t/T-0.5,p) / (2*big(T)^3)
d4_ψ(t,T,p) = d4_f(t/T-0.5,p) / (2*big(T)^4)
d5_ψ(t,T,p) = d5_f(t/T-0.5,p) / (2*big(T)^5)


Tf = 1200;
ps = 10; # steepness
tgrid = 0 : 1 : Tf

dψ1(t) = d1_ψ(t,Tf,ps)
dψ2(t) = d2_ψ(t,Tf,ps)
dψ3(t) = d3_ψ(t,Tf,ps)
dψ4(t) = d4_ψ(t,Tf,ps)
dψ5(t) = d5_ψ(t,Tf,ps)

θinit = 300;
Δr = 200;
ref(t) = θinit + Δr*ψ(t,Tf,ps)

ref_siso = hcat(ref.(tgrid), Δr*dψ1.(tgrid), Δr*dψ2.(tgrid), Δr*dψ3.(tgrid), Δr*dψ4.(tgrid), Δr*dψ5.(tgrid))
u_raw = (Mu*ref_siso')'

# using Plots
# plot(tgrid, u_raw)



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
function loss_optim_L2(u,p)
    input_oc(t) = input_obc(t,[p[1], p[2], u[1]])
    input_error = u_fbc-input_oc.(tgrid)
    err = sum(abs2, input_error) / sum(abs2, u_fbc)
    return err
end


u_fbc = Float64.(max.(u_raw,0))
p1 = Float64(log(maximum(u_fbc)))
t_max = tgrid[argmax(u_fbc)]
p2 = Tf / t_max

p3grid = 1:0.01:20
opt_costs_L2 = mapreduce(p3 -> loss_optim_L2(p3,[p1,p2]),vcat, p3grid)

# plot(p3grid,opt_costs_L2)


# Find optimal p3
using Optimization, OptimizationOptimJL
opt_p = [p1, p2]
opt_u0 = [2.0]
loss_optim_L2(opt_u0, opt_p)
optf_L2 = OptimizationFunction(loss_optim_L2, Optimization.AutoForwardDiff())
opt_prob_L2 = OptimizationProblem(optf_L2, opt_u0, opt_p)

p3_sol = solve(opt_prob_L2, ConjugateGradient())
p3_opt_L2 = p3_sol.u
p3_obj = p3_sol.objective

p_found = [p1, p2,p3_opt_L2[1]]

#=
# λ=60
p_found = [11.81522609265776
            2.070393374741201
            9.212088025161702]

# λ=40
p_found = [11.83256078184801
            2.1052631578947367
            9.38415971553497]            
=#


using CairoMakie

path2folder = "results/figures/controlled/"
filename = path2folder*"feedforward_1d_p3_objective.pdf"
begin   
    fig = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(fig[1, 1], xlabel = L"Parameter $p_{3}$", ylabel = "Objective", 
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    ax1.xticks = 2 : 2 : 20;
    # ax1.yticks = 0 : 2.5 : 10;
    #ax1.yticks = -10 : 5 : 20;
   
    lines!(p3grid, opt_costs_L2;   linestyle = :dash,  linewidth = 5, label=L"$L_{2}$")
    scatter!(p3_opt_L2,[p3_obj], marker = :xcross, markersize=25, color=:purple)
    
    ax2 = Axis(fig, bbox=BBox(306, 572, 170, 330), ylabelsize = 24)
    ax2.xticks = [8,9,10];
    #ax2.yticks = [390, 395,400];
    lines!(ax2, p3grid[700:900], opt_costs_L2[700:900];   linestyle = :dash,  linewidth = 5, color=Makie.wong_colors()[1])
    scatter!(p3_opt_L2,[p3_obj], marker = :xcross, markersize=15, color=:purple)
    translate!(ax2.scene, 0, 0, 10);

    fig
    save(filename, fig, pt_per_unit = 1)       
end

input_u(t) = input_obc(t,p_found)
input_data = input_u.(tgrid)

begin   
    filename = path2folder*"feedforward_1d_fbc_approx_input.pdf"
    fig = Figure(size=(600,400),fontsize=26)

    ax1 = Axis(fig[1, 1], xlabel = "Time t in [s]", ylabel=L"Input $\times 10^4$",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    scale = 1e-4;

    ax1.xticks = 0 : 200 : 1200;
    ax1.yticks = 0 : 2 : 14;

    #tstart = 450;
    #tend = 2550;
    samp1 = 1:100:length(tgrid)
    # lines!(tgrid1[tstart:tend], scale*input_data[tstart:tend,1];   linestyle = :solid, color=Makie.wong_colors()[4] ,   linewidth = 3, label="FBC")
    lines!(tgrid, scale*input_data;   linestyle = :solid,   linewidth = 5, label="Approx.")
    scatter!(tgrid[samp1], scale*u_fbc[samp1]; markersize=20,marker=:diamond, color=Makie.wong_colors()[2] ,  label="FBC")

    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);
      
    fig  
    save(filename, fig, pt_per_unit = 1)   
end



# 1D heat equation
function heat_eq!(dx,x,p,t)       
    u_in =  input_obc(t,p)
    dx .= A*x + B*u_in
end


const ts = Tf/60   # Time step width

# Simulation without optimization
using OrdinaryDiffEq

x0 = 300 * ones(Nx) # Intial values
tspan = (0.0, Tf)   # Time span
alg = KenCarp4()    # Numerical integrator

prob = ODEProblem(heat_eq!,x0,tspan)
sol = solve(prob,alg, p=p_found, saveat = ts)

y_data = Array(sol)[5,:]
ref_data = ref.(sol.t)

begin   
    filename = path2folder*"feedforward_1d_fbc_approx_output.pdf"
    fig = Figure(size=(600,400),fontsize=26)

    ax1 = Axis(fig[1, 1], xlabel = "Time t in [s]", ylabel="Temperature in [K]",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    scale = 1e-4;

    ax1.xticks = 0 : 200 : 1200;
    #ax1.yticks = 0 : 2 : 14;

    #tstart = 450;
    #tend = 2550;
    samp1 = 1:100:length(tgrid)
    lines!(sol.t, y_data;   linestyle = :dash,   linewidth = 5, label="Output")
    lines!(sol.t, ref_data;   linestyle = :dot,   linewidth = 5, label="Reference")
    #scatter!(tgrid[samp1], scale*u_fbc[samp1]; markersize=20,marker=:diamond, color=Makie.wong_colors()[2] ,  label="Reference")

    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);
     
    ax2 = Axis(fig, bbox=BBox(410, 570, 140, 260), ylabelsize = 24)
    ax2.xticks = 800 : 200 : Tf;
    ax2.yticks = [495, 500];
    lines!(sol.t[41:end], y_data[41:end];   linestyle = :dash,   linewidth = 5)
    lines!(sol.t[41:end], ref_data[41:end];   linestyle = :dot,   linewidth = 5)
    #scatter!(sol_final.t[81:20:end], ref_data[1,81:20:end];  markersize = 15, marker = :diamond, color=:black, label = "Reference")
    translate!(ax2.scene, 0, 0, 10);


    fig  
    save(filename, fig, pt_per_unit = 1)   
end