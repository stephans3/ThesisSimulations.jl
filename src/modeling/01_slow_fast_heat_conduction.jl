#=
Figure 2.2: Comparison of slow (above) and fast (below) heat conduction.
=#

using Hestia 

property_slow = StaticIsotropic(5e-6, 1, 1)
property_fast = StaticIsotropic(2e-5, 1, 1)

L = 0.1
N = 20;
heatrod  = HeatRod(L, N)
boundary = Boundary(heatrod)

### Simulation ###
function heat_conduction!(dw, w, param, t)
    dΘ1 = @views dw[1:N]
    dΘ2 = @views dw[N+1:2N]

    Θ1 = @views w[1:N]
    Θ2 = @views w[N+1:2N]

    diffusion!(dΘ1, Θ1, heatrod, property_slow, boundary)
    diffusion!(dΘ2, Θ2, heatrod, property_fast, boundary)
end

θinit = zeros(N)
@. θinit[7:N-6] = 1

using OrdinaryDiffEq
tspan = (0, 20.0)
alg = KenCarp5()
prob= ODEProblem(heat_conduction!,vcat(θinit,θinit),tspan)
sol = solve(prob,alg, saveat=1.0)

xgrid = L/(2N) : L/N : L # Position in x-direction
base_path = "results/figures/modeling/"

using CairoMakie

begin
    data10 =  sol[1][1:N]
    data11 =  sol[11][1:N]
    data12 =  sol[21][1:N]
    
    data20 = sol[1][N+1:2N]
    data21 = sol[11][N+1:2N]
    data22 = sol[21][N+1:2N]
    

    fig1 = Figure(size=(800,600),fontsize=26)
    ax1 = Axis(fig1[1, 1], ylabelsize = 30,
        xlabelsize = 30, xgridstyle = :dash, ygridstyle = :dash, 
        xtickalign = 1., xticksize = 10, 
        xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,
        yminorgridvisible = true, yminorticksvisible = true, yminortickalign = 1,
        ytickalign = 1, yticksize = 10, xlabelpadding = 0)
    
    # ax1.xticks = θgrid[begin] : 50 : θgrid[end];
    # ax1.yticks = -16 : 2 :0; #-16e3 : 2e3 : 0;
    lines!(ax1, xgrid, data10;linestyle = :dot, linewidth = 5, label = "initial")
    lines!(ax1, xgrid, data11;linestyle = :dash, linewidth = 5, label = "proceeding")
    lines!(ax1, xgrid, data12;linestyle = :solid, linewidth = 5, label="final")
    axislegend(ax1 ; position = :rt, backgroundcolor = (:grey90, 0.1));
    
    ax2 = Axis(fig1[2, 1], xlabel = "Length in [m]", ylabelsize = 30,
        xlabelsize = 30, xgridstyle = :dash, ygridstyle = :dash, 
        xtickalign = 1., xticksize = 10, 
        xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,
        yminorgridvisible = true, yminorticksvisible = true, yminortickalign = 1,
        ytickalign = 1, yticksize = 10, xlabelpadding = 0)
    
    # ax1.xticks = θgrid[begin] : 50 : θgrid[end];
    # ax1.yticks = -16 : 2 :0; #-16e3 : 2e3 : 0;
    lines!(ax2, xgrid, data20;linestyle = :dot, linewidth = 5, label = "initial")
    lines!(ax2, xgrid, data21;linestyle = :dash, linewidth = 5, label = "proceeding")
    lines!(ax2, xgrid, data22;linestyle = :solid, linewidth = 5, label="final")
     axislegend(ax2; position = :rt, backgroundcolor = (:grey90, 0.1));
    fig1
    save(base_path*"compare_slow_fast_heat_conduction.pdf", fig1, pt_per_unit = 1)    
end
