
L = 0.1; # Length of 1D rod
# Aluminium
λ = 50;  # Thermal conductivity
ρ = 8000; # Density
c = 400;  # Specific heat capacity
α = λ / (ρ * c) # Diffusivity

Tf = 3000; # Final simulation time

# Reference
Δr = 100
ψ(t) = (1+tanh(10*(t/Tf - 1/2)))/2
ref_init = 300
ref(t) = ref_init + Δr*ψ(t)

using Hestia
prop = StaticIsotropic(λ, ρ, c)

N = 10
rod = HeatRod(L, N)
boundary = Boundary(rod)
actuation = IOSetup(rod)
setIOSetup!(actuation, rod, 1, 1.0,  :west)



# Boundary without Thermal emission (insulated)
boundary_insulated = Boundary(rod) 

# Boundary with Thermal emission
h = 10;
Θamb = 300;
ε = 0.2;
em_total = Emission(h, Θamb,ε) 
boundary_open = Boundary(rod)
setEmission!(boundary_open, em_total, :west);
setEmission!(boundary_open, em_total, :east);

ts = 1
tgrid = 0:ts:Tf

dU = c*ρ*L*Δr; # Δ Internal Energy

E_em_1 = 2*(h*(ts*sum(ref.(tgrid[2:end])) - Tf*Θamb))
coeff_rad = ε*5.67*1e-8;
E_em_2 = 2*(coeff_rad*ts*sum(ref.(tgrid[2:end]).^4))
E_em_approx =  E_em_1 + E_em_2


p2 = 2;
u0 = 1e-4;


using SpecialFunctions
using NLsolve


function input_obc(t,p)
    return exp(p[1] - p[3]^2 * (t/Tf - 1/p[2])^2)
end

u_in_energy(p₁,p₂,p₃) = exp(p₁)*Tf*sqrt(pi) * (erf(p₃-p₃/p₂) - erf(-p₃/p₂)) / (2 * p₃)


function opt_energy_insulated!(F, x)
    u₀ = 1e-4;
    F[1] = x[1] - exp(2x[1]) *π*Tf^2/(p2*dU)^2 - log(u₀)
    F[2] = u₀*sqrt(π)*Tf*exp((x[2]/p2)^2) - x[2]*dU
end

xinit = [10.0, 10.0]
sol_nl_em_approx = nlsolve(opt_energy_insulated!, xinit)
p1_ins,p3_ins = sol_nl_em_approx.zero

p_ins = [p1_ins,p2,p3_ins]
E_in_ins = u_in_energy(p_ins...)

function opt_energy_open!(F, x)
    u₀ = 1e-4;
    F[1] = x[1] - exp(2x[1]) *π*Tf^2/(p2*(dU+E_em_approx))^2 - log(u₀)
    F[2] = u₀*sqrt(π)*Tf*exp((x[2]/p2)^2) - x[2]*(dU+E_em_approx)
end


xinit = [10.0, 10.0]
sol_nl_em_approx = nlsolve(opt_energy_open!, xinit)
p1_open,p3_open = sol_nl_em_approx.zero

p_open = [p1_open,p2,p3_open]
E_in_open = u_in_energy(p_open...)

p1grid = 10.5:0.05:11.2;
p3grid = 8.5:0.05:9.2;

data_loss_insu = zeros(length(p1grid),2);
data_loss_open = zeros(length(p1grid),2);
for (i1,p1) in enumerate(p1grid)
    tempdata = zeros(2)
    opt_energy_insulated!(tempdata, [p1,p3grid[i1]])
    data_loss_insu[i1,:] = copy(tempdata)

    opt_energy_open!(tempdata, [p1,p3grid[i1]])
    data_loss_open[i1,:] = copy(tempdata)
end



using CairoMakie
path2folder = "results/figures/controlled/"
begin   
    data1 = data_loss_insu[:,1]
    data2 = data_loss_open[:,1]

    filename = path2folder*"obc_energy_1d_p1_error.pdf"
    f = Figure(size=(600,400),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel = L"Parameter $p_{1}$", ylabel=L"Error $~$",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    scale = 1;

    ax1.xticks = p1grid[1] : 0.1 : p1grid[end];
    #ax1.yticks = 0 : 2 : 10;

    #tstart = 450;
    #tend = 2550;
    lines!(p1grid, scale*data1;   linestyle = :dot,   linewidth = 5, label="Insulated")
    lines!(p1grid, scale*data2;  linestyle = :dash,  linewidth = 5, label="Emissions")
    scatter!([p1_ins,p1_open],zeros(2), marker = :xcross, markersize=25, color=:purple)
    axislegend(; position = :lc, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f  
    save(filename, f, pt_per_unit = 1)   
end


begin   
    data1 = data_loss_insu[:,2]
    data2 = data_loss_open[:,2]

    filename = path2folder*"obc_energy_1d_p3_error.pdf"
    f = Figure(size=(600,400),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel = L"Parameter $p_{3}$", ylabel=L"Error $\times 10^{8}$",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    scale = 1e-8;

    ax1.xticks = p3grid[1] : 0.1 : p3grid[end];
    #ax1.yticks = 0 : 2 : 10;

    #tstart = 450;
    #tend = 2550;
    lines!(p3grid, scale*data1;   linestyle = :dot,   linewidth = 5, label="Insulated")
    lines!(p3grid, scale*data2;  linestyle = :dash,  linewidth = 5, label="Emissions")
    scatter!([p3_ins,p3_open],zeros(2), marker = :xcross, markersize=25, color=:purple)
    axislegend(; position = :lc, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f  
     save(filename, f, pt_per_unit = 1)   
end


function heat_conduction_open!(dw, w, param, t) 
    u_in1 = ones(1)*input_obc(t,param)

    diffusion!(dw, w, rod, prop, boundary_open,actuation,u_in1)
end

using OrdinaryDiffEq
ref_init = 300;
θinit =  ref_init* ones(N) # Intial values
Tf    =  3000;
tspan =  (0.0, Tf)   # Time span
tsamp = 1.0;
alg = KenCarp4()    # Numerical integrator
prob = ODEProblem(heat_conduction_open!,θinit,tspan)
sol_ins = solve(prob, alg,p=p_ins, saveat = tsamp)
sol_open = solve(prob, alg,p=p_open, saveat = tsamp)

u_in_ins(t) = input_obc(t,p_ins)
input_ins_data = u_in_ins.(sol_ins.t)

u_in_open(t) = input_obc(t,p_open)
input_open_data = u_in_open.(sol_open.t)





begin   
    filename = path2folder*"obc_energy_1d_input_ins_open.pdf"
    f = Figure(size=(600,400),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel = "Time t in [s]", ylabel=L"Input $\times 10^4$",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    scale = 1e-4;

    ax1.xticks = 0 : 500 : 3000;
    ax1.yticks = 0 : 2 : 10;

    #tstart = 450;
    #tend = 2550;
    lines!(sol_ins.t, scale*input_ins_data;   linestyle = :dot,   linewidth = 5, label="Insulated")
    lines!(sol_ins.t, scale*input_open_data;  linestyle = :dash,  linewidth = 5, label="Emissions")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f  
    save(filename, f, pt_per_unit = 1)   
end



begin   
    filename = path2folder*"obc_energy_1d_output_ins_open.pdf"
    θ_ins = sol_ins[end,:]
    θ_open = sol_open[end,:]
    ref_data = ref.(sol_open.t)

    f = Figure(size=(600,400),fontsize=26)

     
    ax2 = Axis(f[1, 1], xlabel = "Time t in [s]", ylabel="Temperature in [K]",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)


    ax2.xticks = 0 : 500 : 3000;
    ax2.yticks = 300 : 25 : 400;
      lines!(sol_open.t, θ_ins; linestyle = :dot,  linewidth = 5, label="Insulated")
    lines!(sol_open.t, θ_open;  linestyle = :dash, linewidth = 5,  label="Emissions")
    scatter!(sol_open.t[1:250:end], ref_data[1:250:end];  markersize = 15, marker = :diamond, color=:black, label = "Reference")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);

    ax2 = Axis(f, bbox=BBox(430, 565, 147, 267), ylabelsize = 24)
    #ax2.xticks = [1450,1500,1550];
    #ax2.yticks = [6,6.1];
    lines!(ax2, sol_open.t[2000:end],θ_ins[2000:end];   linestyle = :dot,  linewidth = 5, color=Makie.wong_colors()[1])
    lines!(ax2, sol_open.t[2000:end],θ_open[2000:end]; linestyle = :dash, linewidth = 5, color=Makie.wong_colors()[2])
    scatter!(sol_open.t[2000:250:end], ref_data[2000:250:end];  markersize = 15, marker = :diamond, color=:black)
    translate!(ax2.scene, 0, 0, 10);

    f  
    save(filename, f, pt_per_unit = 1)   
end




using Plots
plot(p1grid, data_loss_insu[:,1])
plot!(p1grid, data_loss_open[:,1])

plot(p3grid, data_loss_insu[:,2])
plot!(p3grid, data_loss_open[:,2])
