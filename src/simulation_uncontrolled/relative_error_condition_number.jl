using LinearAlgebra

Nx_arr = collect(3:1:100)
e_rel = zeros(length(Nx_arr),2);

for (idx, Nx) in enumerate(Nx_arr)
    D = zeros(Int64,Nx, Nx)
    j=1
    di = (j-1)*Nx
    for i=2:Nx-1
        D[i+di,i-1+di : i+1+di] = [1,-2,1];
    end

    D[1+di,1+di:2+di] = [-1,1]
    D[Nx+di,Nx-1+di:Nx+di] = [1,-1]

    w = collect(1:(1/(Nx-1)):2)
    
    w1 = w / norm(w)

    e_rel[idx,1] = norm(D[:,1]) / norm(D*w1)
    e_rel[idx,2] = norm(D[:,2]) / norm(D*w1)
end

using CairoMakie
base_path = "results/figures/simulation/"

begin
    data1 = e_rel[:,1]
    data2 = e_rel[:,2]

    fig1 = Figure(size=(800,600),fontsize=26)
    ax1 = Axis(fig1[1, 1], xlabel = "Number of Cells", ylabel = "Relative Error", 
        xlabelsize = 30, ylabelsize = 30, xgridstyle = :dash, ygridstyle = :dash, 
        xtickalign = 1., xticksize = 10, 
        xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,
        yminorgridvisible = true, yminorticksvisible = true, yminortickalign = 1,
        ytickalign = 1, yticksize = 10, xlabelpadding = 0)
    
    ax1.xticks = vcat(3, collect(20:20:100));
    ax1.yticks = 0 : 500 : 2500;
  
    lines!(ax1, Nx_arr, data1;linestyle = :dot, linewidth = 5, label = "Rel. error 1")
    lines!(ax1, Nx_arr, data2;linestyle = :dash, linewidth = 5, label = "Rel. error 2")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1));
    fig1
    save(base_path*"condition_relative_error_1d.pdf", fig1, pt_per_unit = 1)    
end