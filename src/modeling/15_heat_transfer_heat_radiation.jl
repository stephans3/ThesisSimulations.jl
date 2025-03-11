using Hestia
prop = StaticIsotropic(50, 8000, 400)

L = 0.2
N = 40
rod = HeatRod(L, N)

boundary1 = Boundary(rod)
boundary2 = Boundary(rod)
boundary3 = Boundary(rod)

h = 10;
Θamb = 300;
ε = 0.2;
em_transfer = Emission(h, Θamb) # Pure heat transfer
em_radiation = Emission(0, 0,ε) # Pure heat radiation
em_total = Emission(h, Θamb,ε) # Pure heat radiation

setEmission!(boundary1, em_transfer, :east);
setEmission!(boundary2, em_radiation, :east);
setEmission!(boundary3, em_total, :east);

function heat_conduction_tr!(dw, w, param, t)
    diffusion!(dw, w, rod, prop, boundary1)
end

function heat_conduction_rad!(dw, w, param, t)
    diffusion!(dw, w, rod, prop, boundary2)
end

function heat_conduction_total!(dw, w, param, t)
    diffusion!(dw, w, rod, prop, boundary3)
end


using OrdinaryDiffEq
θinit = 600ones(N) 
tspan = (0.0, 2500)
tsave = 10.0;
alg = KenCarp5()
prob_tr = ODEProblem(heat_conduction_tr!,θinit,tspan)
prob_rad = ODEProblem(heat_conduction_rad!,θinit,tspan)
prob_total = ODEProblem(heat_conduction_total!,θinit,tspan)

sol_tr = solve(prob_tr,alg, saveat=tsave)
sol_rad = solve(prob_rad,alg, saveat=tsave)
sol_total = solve(prob_total,alg, saveat=tsave)

data_tr = sol_tr[end,:]
data_rad = sol_rad[end,:]
data_total = sol_total[end,:]

tgrid = 0 : tsave : tspan[2]
# xgrid = L/(2N) : L/N : L # Position in x-direction
base_path = "results/figures/modeling/"

using CairoMakie

begin
    fig1 = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(fig1[1, 1], xlabel = "Time in [s]", ylabel = "Temperature in [K]", 
        xlabelsize = 30, ylabelsize = 30, xgridstyle = :dash, ygridstyle = :dash, 
        xtickalign = 1., xticksize = 10, 
        xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,
        yminorgridvisible = true, yminorticksvisible = true, yminortickalign = 1,
        ytickalign = 1, yticksize = 10, xlabelpadding = 0, limits = (nothing, (570, 601)))
    
    ax1.xticks = 0: 500 : tspan[2];
    ax1.yticks = 570: 5 : 600;
  
    lines!(ax1, tgrid, data_tr;linestyle = :dot, linewidth = 5, label = "Only heat transfer")
    lines!(ax1, tgrid, data_rad;linestyle = :dash, linewidth = 5, label = "Only heat radiation")
    lines!(ax1, tgrid, data_total;linestyle = :solid, linewidth = 5, label="Heat transfer and radiation")
    axislegend(; position = :lb, backgroundcolor = (:grey90, 0.1));
    fig1
    save(base_path*"heat_transfer_radiation_by_time.pdf", fig1, pt_per_unit = 1)    
end

θgrid = collect(300: 1.0 : 700);
Phi_trans_data = similar(θgrid)
Phi_rad_data = similar(θgrid)
Phi_total_data = similar(θgrid)

emit!(Phi_trans_data, θgrid, em_transfer)
emit!(Phi_rad_data, θgrid, em_radiation)
emit!(Phi_total_data, θgrid, em_total)

Phi_trans_data *= 10^(-3) 
Phi_rad_data *= 10^(-3)  
Phi_total_data *= 10^(-3) 

begin
    fig1 = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(fig1[1, 1], xlabel = "Temperature in [K]", ylabel = L"Emissions $\times 10^{3}$ in $\left[\frac{W}{m^2}\right]$",
        xlabelsize = 30, ylabelsize = 30, xgridstyle = :dash, ygridstyle = :dash, 
        xtickalign = 1., xticksize = 10, 
        xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,
        yminorgridvisible = true, yminorticksvisible = true, yminortickalign = 1,
        ytickalign = 1, yticksize = 10, xlabelpadding = 0,limits = (nothing, (-9, 0.5)))
    
    ax1.xticks = θgrid[begin] : 100 : θgrid[end];
    ax1.yticks = -8 : 2 :0; #-16e3 : 2e3 : 0;
    lines!(θgrid, Phi_trans_data;linestyle = :dot, linewidth = 5, label = "Only heat transfer")
    lines!(θgrid, Phi_rad_data;linestyle = :dash, linewidth = 5, label="Only heat radiation")
    lines!(θgrid, Phi_total_data; linewidth = 5, label="Heat transfer and radiation")
    axislegend(; position = :lb, backgroundcolor = (:grey90, 0.1));
    fig1
    save(base_path*"heat_transfer_radiation_by_temperature.pdf", fig1, pt_per_unit = 1)    
end

