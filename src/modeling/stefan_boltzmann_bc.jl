
# Heat transfer coefficient
h = 5

# Heat radiation
ε = 0.2 # Emissivity
sb = 5.6703744191844294e-8 # Stefan-Boltzmann constant
k = ε * sb 
θamb = 300;

θgrid = 200: 1 : 700;

Φ_trans(θ) = -h * (θ - θamb)
Φ_rad(θ) = -k * θ^4
Φem(θ) = -h * (θ - θamb) - k * θ^4

Phi_trans_data = Φ_trans.(θgrid)
Phi_rad_data = Φ_rad.(θgrid)
Phi_total_data =  Φem.(θgrid)

Phi_trans_data *= 10^(-3) 
Phi_rad_data *= 10^(-3)  
Phi_total_data *= 10^(-3) 

base_path = "results/figures/modeling/"
using CairoMakie

begin
    fig1 = Figure(size=(800,600),fontsize=20)
    ax1 = Axis(fig1[1, 1], xlabel = "Temperature in [K]", ylabel = L"Emissions $\times 10^{3}$ in $\left[\frac{W}{m^2}\right]$", ylabelsize = 22,
        xlabelsize = 24, xgridstyle = :dash, ygridstyle = :dash, 
        xtickalign = 1., xticksize = 10, 
        xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,
        yminorgridvisible = true, yminorticksvisible = true, yminortickalign = 1,
        ytickalign = 1, yticksize = 10, xlabelpadding = 0)
    
    ax1.xticks = θgrid[begin] : 50 : θgrid[end];
    ax1.yticks = -16 : 2 :0; #-16e3 : 2e3 : 0;
    lines!(θgrid, Phi_trans_data;linestyle = :dot, linewidth = 3, label = "Only heat transfer")
    lines!(θgrid, Phi_rad_data;linestyle = :dash, linewidth = 3, label="Only heat radiation")
    lines!(θgrid, Phi_total_data; linewidth = 3, label="Heat transfer and radiation")
    axislegend(; position = :lb, backgroundcolor = (:grey90, 0.1));
    fig1
    #save(base_path*"stefan_boltzmann_bc.pdf", fig1, pt_per_unit = 1)    
end

