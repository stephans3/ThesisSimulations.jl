
θ_latex = collect(3:1:7)
θ = collect(300:100:700)
λdata = [40, 50, 70, 85, 90]

function findparameters(x, y)

    n = length(x)
    M = zeros(BigFloat,length(x), length(x))

    for i=1 : n
        for j=1 :n
            M[i,j] = x[i]^(j-1)
        end
    end
    
    pars = inv(M)*y
    return pars 
end

p1 = findparameters(θ,λdata)
λapprox = [370, -2.85, 8.458e-3, -1e-5, 4.1667e-9]

p_latex = findparameters(θ_latex,λdata)




using Hestia
prop_dynamic = DynamicIsotropic(λapprox, [8000], [400])

L = 0.2
N = 40
rod = HeatRod(L, N)
boundary = Boundary(rod)

function heat_conduction_dynamic!(dw, w, param, t)
    diffusion!(dw, w, rod, prop_dynamic, boundary)
end


using OrdinaryDiffEq
θinit = 300ones(N)
θinit[1:round(Int,N/2)] .= 700 
tspan = (0.0, 800)
tsave = 2.0;
alg = KenCarp5()
prob_dynamic = ODEProblem(heat_conduction_dynamic!,θinit,tspan)
sol_dynamic = solve(prob_dynamic, alg, saveat=tsave)


tgrid = 0 : tsave : tspan[2]
base_path = "results/figures/modeling/"

using CairoMakie

begin
    data1 = sol_dynamic[1,:]
    data2 = sol_dynamic[11,:]
    data3 = sol_dynamic[21,:]
    data4 = sol_dynamic[31,:]
    data5 = sol_dynamic[40,:]

    fig1 = Figure(size=(600,400),fontsize=24)
    ax1 = Axis(fig1[1, 1], xlabel = "Time in [s]", ylabel = "Temperature in [K]", 
        xlabelsize = 30, ylabelsize = 30, xgridstyle = :dash, ygridstyle = :dash, 
        xtickalign = 1., xticksize = 10, 
        xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,
        yminorgridvisible = true, yminorticksvisible = true, yminortickalign = 1,
        ytickalign = 1, yticksize = 10, xlabelpadding = 0,limits = (nothing, (270, 810)))
    
    ax1.xticks = 0: 100 : tspan[2];
    ax1.yticks = 300: 100 : 800;
  
    lines!(ax1, tgrid, data1;linestyle = :solid, linewidth = 5, label = "x=0")
    lines!(ax1, tgrid, data2;linestyle = :dash, linewidth = 5, label = "x=L/4")
    lines!(ax1, tgrid, data3;linestyle = :dot, linewidth = 5, label = "x=L/2")
    lines!(ax1, tgrid, data4;linestyle = :dash, linewidth = 5, label = "x=3/4 L")
    lines!(ax1, tgrid, data5;linestyle = :solid, linewidth = 5, label="x=L")
    axislegend(; position = :rt, backgroundcolor = (:grey90, 0.1));
    fig1
    save(base_path*"dynamic_heat_conduction_position.pdf", fig1, pt_per_unit = 1)    
end


xgrid = L/(2N) : L/N : L # Position in x-direction
begin
    data1 = sol_dynamic[:,1]
    data2 = sol_dynamic[:,26]
    data3 = sol_dynamic[:,51]
    data4 = sol_dynamic[:,101]
    data5 = sol_dynamic[:,end]

    fig1 = Figure(size=(600,400),fontsize=24)
    ax1 = Axis(fig1[1, 1], xlabel = "Position x in [cm]", ylabel = "Temperature in [K]", 
        xlabelsize = 30, ylabelsize = 30, xgridstyle = :dash, ygridstyle = :dash, 
        xtickalign = 1., xticksize = 10, 
        xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,
        yminorgridvisible = true, yminorticksvisible = true, yminortickalign = 1,
        ytickalign = 1, yticksize = 10, xlabelpadding = 0,limits = (nothing, (270, 810)))
    xscale = 100;
    ax1.xticks = 0: xscale*(L/4) : xscale*L;
    ax1.yticks = 300: 100 : 800;
    lines!(ax1, xscale*xgrid, data1;linestyle = :dot, linewidth = 5, label = "t=0 s")
    lines!(ax1, xscale*xgrid, data2;linestyle = :solid, linewidth = 5, label = "t=50 s")
    lines!(ax1, xscale*xgrid, data3;linestyle = :dash, linewidth = 5, label = "t=100 s")
    lines!(ax1, xscale*xgrid, data4;linestyle = :solid, linewidth = 5, label = "t=200 s")
    lines!(ax1, xscale*xgrid, data5;linestyle = :dash, linewidth = 5, label="t=800 s")
    axislegend(; position = :rt, backgroundcolor = (:grey90, 0.1));
    fig1
    save(base_path*"dynamic_heat_conduction_time.pdf", fig1, pt_per_unit = 1)    
end


#=
θgrid = collect(300: 1.0 : 700);
λ_fun(θ) = specifyproperty(θ,λapprox)
λdata = λ_fun.(θgrid)

begin
    fig1 = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(fig1[1, 1], ylabelsize = 30, xlabel = "Temperature in [K]", ylabel = "Thermal conductivity", 
        xlabelsize = 30, xgridstyle = :dash, ygridstyle = :dash, 
        xtickalign = 1., xticksize = 10, 
        xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,
        yminorgridvisible = true, yminorticksvisible = true, yminortickalign = 1,
        ytickalign = 1, yticksize = 10, xlabelpadding = 0)
    
    ax1.xticks = 300: 50 : 700;
    ax1.yticks = 40 : 5 : 90;
  
    lines!(ax1, θgrid, λdata;linestyle = :dot, linewidth = 5)
    fig1
    save(base_path*"dynamic_heat_conduction_lambda.pdf", fig1, pt_per_unit = 1)    
end
=#