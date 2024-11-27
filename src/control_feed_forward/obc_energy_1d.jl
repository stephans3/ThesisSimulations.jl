


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
ref(t) = ref_init + Δr*ψ(t)


dU = c*ρ*L*Δr; # Δ Internal Energy


using SpecialFunctions
using NLsolve


function input_obc(t,p)
    return exp(p[1] - p[3]^2 * (t/Tf - 1/p[2])^2)
end

u_in_energy(p₁,p₂,p₃) = exp(p₁)*Tf*sqrt(pi) * (erf(p₃-p₃/p₂) - erf(-p₃/p₂)) / (2 * p₃)

function energy_insulated!(F, x)
    u₀ = 1e-4;
    p₂ = p2
    F[1] = u₀ -input_obc(0,[x[1], p₂, x[2]])
    F[2] = dU - u_in_energy(x[1], p₂, x[2]) 
end


p2 = 2;
Finit = zeros(2);
xinit = [10.0, 10.0]
energy_insulated!(Finit,xinit)

sol_p_nl = nlsolve(energy_insulated!, xinit)
p1,p3 = sol_p_nl.zero

round.([p1,p3],digits=3)

p1grid = 10.5:0.05:11.5;
p3grid = 8.5:0.05:9.5;
loss2d = zeros(length(p1grid),length(p3grid),2)

tempdata = zeros(2)
energy_insulated!(tempdata, [p1,p3])

for (i1, p11) in enumerate(p1grid), (i2,p33) in enumerate(p3grid)
    tempdata = zeros(2)
    energy_insulated!(tempdata, [p11,p33])
    loss2d[i1,i2,:] = copy(tempdata)
end

# using Plots
# heatmap(loss2d[:,:,1])
# heatmap(loss2d[:,:,2])



using CairoMakie
path2folder = "results/figures/controlled/"
filename = path2folder*"obc_energy_1d_contour_init_u.pdf"
begin
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = L"Gain $p_{1}$", ylabel = L"Kurtosis $p_{3}$", xlabelsize = 30, ylabelsize = 30)
    data = 1e3*loss2d[:,:,1]
    #tightlimits!(ax1)
    #hidedecorations!(ax1)
    # co = contourf!(ax1, p1grid, p2grid, loss2d, levels=20, colormap=:Greens) #levels = range(0.0, 10.0, length = 20))
    co = contourf!(ax1, p1grid, p3grid, data, levels=20, colormap=:managua) #levels = range(0.0, 10.0, length = 20))
    #lines!(p12_path[1:20,1],p12_path[1:20,2], linestyle = :dash,  linewidth = 5, color=:black)#RGBf(0.5, 0.2, 0.8))
    #lines!(p12_path[1:end,1],p12_path[1:end,2], linestyle = :dash,  linewidth = 5, color=:black)#RGBf(0.5, 0.2, 0.8))
    scatter!([p1],[p3], marker = :xcross, markersize=25, color=:purple)
    #ax1.xticks = 11.22 : 0.04 : 11.34 #[11.2, 11.24, 11.28, 11.32, 11.36];
    #ax1.yticks = 2.05:0.05:2.2;
    Colorbar(f[1, 2], co,label = L"Error $\times 10^{3}$")
    f    

    save(filename, f, pt_per_unit = 1)   
end

filename = path2folder*"obc_energy_1d_contour_energy.pdf"
begin
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = L"Gain $p_{1}$", ylabel = L"Kurtosis $p_{3}$", xlabelsize = 30, ylabelsize = 30)
    data = 1e-7*loss2d[:,:,2]
    #tightlimits!(ax1)
    #hidedecorations!(ax1)
    # co = contourf!(ax1, p1grid, p2grid, loss2d, levels=20, colormap=:Greens) #levels = range(0.0, 10.0, length = 20))
    co = contourf!(ax1, p1grid, p3grid, data, levels=20, colormap=:managua) #levels = range(0.0, 10.0, length = 20))
    scatter!([p1],[p3], marker = :xcross, markersize=25, color=:purple)
    #ax1.xticks = 11.22 : 0.04 : 11.34 #[11.2, 11.24, 11.28, 11.32, 11.36];
    #ax1.yticks = 2.05:0.05:2.2;
    Colorbar(f[1, 2], co,label = L"Error $\times 10^{-7}$")
    f    

    save(filename, f, pt_per_unit = 1)   
end


function heat_conduction!(dw, w, param, t) 
    u_in1 = ones(1)*input_obc(t,param)

    diffusion!(dw, w, rod, prop, boundary,actuation,u_in1)
end

using Hestia
prop = StaticIsotropic(λ, ρ, c)

N = 10
rod = HeatRod(L, N)
boundary = Boundary(rod)
actuation = IOSetup(rod)
setIOSetup!(actuation, rod, 1, 1.0,  :west)

using OrdinaryDiffEq
ref_init = 300;
θinit =  ref_init* ones(N) # Intial values
Tf    =  3000;
tspan =  (0.0, Tf)   # Time span
tsamp = 1.0;
p_orig= [p1,p2,p3]
alg = KenCarp4()    # Numerical integrator
prob = ODEProblem(heat_conduction!,θinit,tspan,p_orig)
sol = solve(prob, alg, saveat = tsamp)

u_in(t) = input_obc(t,p_orig)
input_data = u_in.(sol.t)




using CairoMakie
path2folder = "results/figures/controlled/"

filename = path2folder*"obc_energy_1d_input.pdf"
begin   
    θ_left = sol[1,:]
    θ_center = sol[round(Int64,N/2),:]
    θ_right = sol[end,:]
    f = Figure(size=(600,400),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel = "Time t in [s]", ylabel=L"Input $\times 10^4$",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    scale = 1e-4;

    ax1.xticks = 0 : 500 : 3000;
    ax1.yticks = 0 : 2.5 : 10;

    #tstart = 450;
    #tend = 2550;
    lines!(sol.t, scale*input_data;   linestyle = :dot,   linewidth = 5, label="Initial")
    
    f  
    save(filename, f, pt_per_unit = 1)   
end

filename = path2folder*"obc_energy_1d_temp_evol.pdf"
begin   
    θ_left = sol[1,:]
    θ_center = sol[round(Int64,N/2),:]
    θ_right = sol[end,:]
    f = Figure(size=(600,400),fontsize=26)

     
    ax2 = Axis(f[1, 1], xlabel = "Time t in [s]", ylabel="Temperature in [K]",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)


    ax2.xticks = 0 : 500 : 3000;
    ax2.yticks = 300 : 25 : 400;
    lines!(sol.t, θ_left;   linestyle = :dot,   linewidth = 5, label="Left")
    lines!(sol.t, θ_center; linestyle = :dash,  linewidth = 5, label="Center")
    lines!(sol.t, θ_right;  linestyle = :solid, linewidth = 5, label="Right")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f  
    save(filename, f, pt_per_unit = 1)   
end




# Boundary with Thermal emission
h = 10;
Θamb = 300;
ε = 0.2;
em_total = Emission(h, Θamb,ε) 
boundary_open = Boundary(rod)
setEmission!(boundary_open, em_total, :west);
setEmission!(boundary_open, em_total, :east);

# Emitted heat flux
function phi_em(θ)
    coeff_rad = ε*5.67*1e-8;
    return -h*(θ-Θamb) - coeff_rad*θ^4
end

θ1 = sol[1,2:end]
θ2 = sol[end,2:end]
# Energy of emitted heat flux
E_phi_em = -tsamp*(sum(phi_em.(θ1) + phi_em.(θ2)))

function energy_phi_em!(F, x)
    u₀ = 1e-4;
    p₂ = p2
    F[1] = input_obc(0,[x[1], p₂, x[2]]) - u₀
    F[2] = E_phi_em + dU - u_in_energy(x[1], p₂, x[2])
end

sol_nl_phi_em = nlsolve(energy_phi_em!, xinit)
p11,p31 = sol_nl_phi_em.zero

round.([p11,p31],digits=3)

# Approximation of the emitted heat flux

E_em_1 = 2*(h*(tsamp*sum(ref.(sol.t[2:end])) - Tf*Θamb))
coeff_rad = ε*5.67*1e-8;
E_em_2 = 2*(coeff_rad*tsamp*sum(ref.(sol.t[2:end]).^4))
E_em_approx =  E_em_1 + E_em_2

function energy_em_approx!(F, x)
    u₀ = 1e-4;
    p₂ = p2
    F[1] = u₀- input_obc(0,[x[1], p₂, x[2]])
    F[2] = E_em_approx + dU - u_in_energy(x[1], p₂, x[2])
end


sol_nl_em_approx = nlsolve(energy_em_approx!, xinit)
p12,p32 = sol_nl_em_approx.zero

round.([p12,p32],digits=3)

# Simulation with thermal emissions
function heat_conduction_open!(dw, w, param, t) 
    u_in1 = ones(1)*input_obc(t,param)

    diffusion!(dw, w, rod, prop, boundary_open,actuation,u_in1)
end


prob_em = ODEProblem(heat_conduction_open!,θinit,tspan)

p_phi_em = abs.([p11,p2,p31])
p_em_approx = abs.([p12,p2,p32])

sol_phi_em      = solve(prob_em, alg, p=p_phi_em, saveat = tsamp)
sol_em_approx   = solve(prob_em, alg, p=p_em_approx, saveat = tsamp)

u_in_1(t) = input_obc(t,p_phi_em)
u_in_2(t) = input_obc(t,p_em_approx)
input_data_1 = u_in_1.(sol.t)
input_data_2 = u_in_2.(sol.t)


filename = path2folder*"obc_energy_1d_emission_input.pdf"
begin   
    f = Figure(size=(600,400),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel = "Time t in [s]", ylabel=L"Input $\times 10^4$",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    scale = 1e-4;

    ax1.xticks = 0 : 500 : 3000;
    #ax1.yticks = 0 : 2.5 : 10;

    #tstart = 450;
    #tend = 2550;
    
    err = input_data_1 - input_data_2
    #lines!(sol.t, err;   linestyle = :dot,   linewidth = 5) #, label="Real Emissions")
    lines!(sol.t, scale*input_data_1;   linestyle = :dot,   linewidth = 5, label="True Em.")
    lines!(sol.t, scale*input_data_2;   linestyle = :dash,   linewidth = 5, label="Approx. Em.")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);
    

    #ax2 = Axis(f, bbox=BBox(170, 330, 250, 360), ylabelsize = 24)
    ax2 = Axis(f, bbox=BBox(415, 560, 255, 370), ylabelsize = 24)
    ax2.xticks = [1450,1500,1550];
    ax2.yticks = [6,6.1];
    lines!(ax2, sol.t[1450:1550],scale*input_data_1[1450:1550];   linestyle = :dot,  linewidth = 5, color=Makie.wong_colors()[1])
    lines!(ax2, sol.t[1450:1550],scale*input_data_2[1450:1550]; linestyle = :dash, linewidth = 5, color=Makie.wong_colors()[2])
    translate!(ax2.scene, 0, 0, 10);
    
    f  
    save(filename, f, pt_per_unit = 1)   
end



filename = path2folder*"obc_energy_1d_emission_output.pdf"
begin   
    y_data_phi_em = sol_phi_em[end,:]
    y_data_em_approx = sol_em_approx[end,:]
    
    f = Figure(size=(600,400),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel = "Time t in [s]", ylabel="Temperature in [K]",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    scale = 1e-4;

    ax1.xticks = 0 : 500 : 3000;
    ax1.yticks = 300 : 25 : 400;

    tstart = 450;
    tend = 2550;
    lines!(sol.t, y_data_phi_em;   linestyle = :dot,   linewidth = 5, label="True Em.")
    lines!(sol.t, y_data_em_approx;   linestyle = :dash,   linewidth = 5, label="Approx. Em.")
    #lines!(sol_orig.t, y_data_adjust;   linestyle = :dash,   linewidth = 5, label="Adjusted")
    #scatter!(sol_orig.t[1:250:end], ref_data[1:250:end];  markersize = 15, marker = :diamond, color=:black, label = "Reference")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);

    ax2 = Axis(f, bbox=BBox(420, 560, 140, 250), ylabelsize = 24)
    ax2.xticks = 2000 : 500 : Tf;
    ax2.yticks = [400,405];
    lines!(ax2, sol_phi_em.t[2000:end],y_data_phi_em[2000:end];   linestyle = :dot,  linewidth = 5, color=Makie.wong_colors()[1])
    lines!(ax2, sol_phi_em.t[2000:end],y_data_em_approx[2000:end]; linestyle = :dash, linewidth = 5, color=Makie.wong_colors()[2])
    translate!(ax2.scene, 0, 0, 10);

    f  
    save(filename, f, pt_per_unit = 1)   
end
