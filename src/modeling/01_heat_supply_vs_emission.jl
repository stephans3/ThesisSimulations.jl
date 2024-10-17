using Hestia
prop = StaticIsotropic(50, 8000, 400)

L = 0.2
N = 40
rod = HeatRod(L, N)

boundary = Boundary(rod)

actuation = IOSetup(rod)
setIOSetup!(actuation, rod, 1, 1.0,  :west)
setIOSetup!(actuation, rod, 2, 1.0,  :east)

function heat_conduction!(dw, w, param, t)
    dΘ1 = @views dw[1:N]
    dΘ2 = @views dw[N+1:2N]

    Θ1 = @views w[1:N]
    Θ2 = @views w[N+1:2N]
    
    u_in1 = [1e4,-1e4]
    u_in2 = [2e4,-1e4]

    diffusion!(dΘ1, Θ1, rod, prop, boundary,actuation,u_in1)
    diffusion!(dΘ2, Θ2, rod, prop, boundary,actuation,u_in2)
end

using OrdinaryDiffEq
θinit = 300ones(N) 
tspan = (0.0, 300.0)
alg = KenCarp5()
prob = ODEProblem(heat_conduction!,vcat(θinit,θinit),tspan)
sol = solve(prob,alg, saveat=30.0)

xgrid = L/(2N) : L/N : L # Position in x-direction
base_path = "results/figures/modeling/"

using CairoMakie

begin
    data10 =  sol[1][1:N]
    data11 =  sol[6][1:N]
    data12 =  sol[end][1:N]
    
    data20 = sol[1][N+1:2N]
    data21 = sol[6][N+1:2N]
    data22 = sol[end][N+1:2N]
    

    fig1 = Figure(size=(800,600),fontsize=26)
    ax1 = Axis(fig1[1, 1],
        xlabelsize = 30, ylabelsize = 30, xgridstyle = :dash, ygridstyle = :dash, 
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
    
    ax2 = Axis(fig1[2, 1], xlabel = "Length in [m]",
        xlabelsize = 30, ylabelsize = 30, xgridstyle = :dash, ygridstyle = :dash, 
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
    save(base_path*"heat_supply_vs_emission.pdf", fig1, pt_per_unit = 1)    
end
