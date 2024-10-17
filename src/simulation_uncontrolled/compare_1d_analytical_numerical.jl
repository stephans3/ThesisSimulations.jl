

function temp_evol(t,x;kmax=100)
    c0 = p*L^2 / 6
    c1 = p*L^2 / pi^2

    s = 0;

    for k=1 : kmax
        s += exp(-4*α*(k*π/L)^2*t) *cos(2*k*pi*x/L) / k^2;
    end

    return c0 - c1*s
end


θ₀ = 300.0  # Initial temperature
λ = 45.0    # Thermal conductivity: constant
ρ = 7800.0  # Mass density: constant
c = 480.0   # Specific heat capacity: constan
α = λ/(ρ*c) # Diffusivity

L = 0.2  # Length
p = 3e4 # 1e5;
temp_init(x) = p*x*(L-x);

Tf = 200;
t_samp = 1.0;
tspan = (0.0, Tf)
tgrid = 0 : t_samp : Tf;

function heat_conduction_rod!(dθ, θ, param, t,rod1d)
    diffusion!(dθ, θ, rod1d, property, Boundary(rod1d))
end

using Hestia, OrdinaryDiffEq
property = StaticIsotropic(λ, ρ, c)
alg = KenCarp5()

Nx_arr = [10, 20, 40, 60, 80, 100]
err_final_arr = zeros(length(Nx_arr));
err_rel_final_arr = zeros(length(Nx_arr));


for (jn, Nx) in enumerate(Nx_arr)
    xgrid = L/(2Nx) : L/Nx : L # Position in x-direction

    # Analytical solution
    temp_analytical = zeros(length(xgrid), length(tgrid))
    for (i,td) in enumerate(tgrid)
        for (k,xd) in enumerate(xgrid)
            temp_analytical[k,i] = θ₀ + temp_evol(td,xd)
        end
    end

    # Numerical solution
    θinit = θ₀ .+ temp_init.(xgrid)
    rod  = HeatRod(L, Nx) # Numerical
    heat_conduction!(dx,x,p,t) = heat_conduction_rod!(dx, x, p, t,rod)
    prob = ODEProblem(heat_conduction!,θinit,tspan)
    sol = solve(prob,alg, saveat=t_samp)
    temp_numerical = Array(sol);

    err_data = temp_analytical - temp_numerical
    err_rel_data = temp_analytical ./ temp_numerical .- 1

    # Absolute error at t=T
    err_final = sum(err_data[:,end])/Nx

    # Relative error per mille at t=T
    err_rel_final = sum(err_rel_data[:,end])/Nx *1000

    err_final_arr[jn] = err_final
    err_rel_final_arr[jn] = err_rel_final  
end

round.(err_final_arr,digits=4)
round.(err_rel_final_arr,digits=4)

#=
p=1e4
Nx         | 10      | 20      | 40      | 60      | 80      | 100
---------------------------------------------------------------------
e_abs      | -0.3333 | -0.0833 | -0.0208 | -0.0093 | -0.0052 | -0.0033
---------------------------------------------------------------------
e_rel in ‰ | -0.9048 | -0.2264 | -0.0566 | -0.0252 | -0.0142 | -0.0091
=#


#=
p=2e4
Nx         | 10      | 20      | 40      | 60      | 80      | 100
---------------------------------------------------------------------
e_abs      | -0.6667 | -0.1667 | -0.0417 | -0.0185 | -0.0104 | -0.0067
---------------------------------------------------------------------
e_rel in ‰ | -1.5264 | -0.3821 | -0.0956 | -0.0425 | -0.0239 | -0.0153
=#


#=
p=3e4
Nx         | 10      | 20      | 40      | 60      | 80      | 100
---------------------------------------------------------------------
e_abs      | -1.0000 | -0.2500 | -0.0625 | -0.0278 | -0.0156 | -0.0100
---------------------------------------------------------------------
e_rel in ‰ | -1.9798 | -0.4959 | -0.1240 | -0.0551 | -0.0310 | -0.0198
=#

e_abs_data = [-0.3333  -0.0833  -0.0208  -0.0093  -0.0052  -0.0033;
              -0.6667  -0.1667  -0.0417  -0.0185  -0.0104  -0.0067;
              -1.0000  -0.2500  -0.0625  -0.0278  -0.0156  -0.0100]

e_rel_data = [-0.9048  -0.2264  -0.0566  -0.0252  -0.0142  -0.0091;
              -1.5264  -0.3821  -0.0956  -0.0425  -0.0239  -0.0153;
              -1.9798  -0.4959  -0.1240  -0.0551  -0.0310  -0.0198]


base_path = "results/figures/simulation_uncontrolled/"
using CairoMakie
begin
    data11 = log.(abs.(e_abs_data[1,:]))
    data21 = log.(abs.(e_abs_data[2,:]))
    data31 = log.(abs.(e_abs_data[3,:]))

    data12 = log.(abs.(e_rel_data[1,:]))
    data22 = log.(abs.(e_rel_data[2,:]))
    data32 = log.(abs.(e_rel_data[3,:]))

    fig1 = Figure(size=(800,600),fontsize=20)
    ax1 = Axis(fig1[1, 1], ylabelsize = 22, xlabel = "Number of Discretization Nodes", ylabel = L"Logarithmic Error $\log(|e|)$", 
        xlabelsize = 24, xgridstyle = :dash, ygridstyle = :dash, 
        xtickalign = 1., xticksize = 10, 
        xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,
        yminorgridvisible = true, yminorticksvisible = true, yminortickalign = 1,
        ytickalign = 1, yticksize = 10, xlabelpadding = 0)
    
    ax1.xticks = Nx_arr;
    ax1.yticks = -6: 1 : 1;
  
    scatterlines!(ax1, Nx_arr, data11; linestyle = :solid, marker=:circle, markersize=15, linewidth = 3, label = L"$e_{abs}$, $p=10^4$")
    scatterlines!(ax1, Nx_arr, data21; linestyle = :dash,  marker=:circle, markersize=15, linewidth = 3, label = L"$e_{abs}$, $p=2\cdot10^4$")
    scatterlines!(ax1, Nx_arr, data31; linestyle = :dot,   marker=:circle, markersize=15, linewidth = 3, label = L"$e_{abs}$, $p=3\cdot10^4$")
    scatterlines!(ax1, Nx_arr, data12; linestyle = :solid, marker=:xcross, markersize=15, linewidth = 3, label = L"$e_{rel}$, $p=10^4$")
    scatterlines!(ax1, Nx_arr, data22; linestyle = :dash,  marker=:xcross, markersize=15, linewidth = 3, label = L"$e_{rel}$, $p=2\cdot10^4$")
    scatterlines!(ax1, Nx_arr, data32; linestyle = :dot,   marker=:xcross, markersize=15, linewidth = 3, label = L"$e_{rel}$, $p=3\cdot10^4$")
    
    axislegend(; position = :rt, backgroundcolor = (:grey90, 0.1));
    fig1
    save(base_path*"abs_rel_error_scatter.pdf", fig1, pt_per_unit = 1)    
end




Nx = 40

xgrid = L/(2Nx) : L/Nx : L # Position in x-direction

# Analytical solution
temp_analytical = zeros(length(xgrid), length(tgrid))
for (i,td) in enumerate(tgrid)
    for (k,xd) in enumerate(xgrid)
        temp_analytical[k,i] = θ₀ + temp_evol(td,xd)
    end
end


# Numerical solution
θinit = θ₀ .+ temp_init.(xgrid)
rod  = HeatRod(L, Nx) # Numerical
heat_conduction!(dx,x,p,t) = heat_conduction_rod!(dx, x, p, t,rod)
prob = ODEProblem(heat_conduction!,θinit,tspan)
sol = solve(prob,alg, saveat=t_samp)
temp_numerical = Array(sol);

err_data = temp_analytical - temp_numerical
err_rel_data = temp_analytical ./ temp_numerical .- 1

err_rel_data[:,end]

base_path = "results/figures/simulation_uncontrolled/"
using CairoMakie
begin      
    f = Figure(size=(600,450),fontsize=20)
    ax1 = Axis(f[1, 1],xlabel = "Position x in [m]", ylabel = "Time t in [s]")
    tightlimits!(ax1)
    #hidedecorations!(ax1)
    lmin = floor(minimum(err_data),digits=1)
    lmax = ceil(maximum(err_data),digits=1)

    co = contourf!(ax1, xgrid,tgrid, err_data, levels = range(lmin, lmax, length = 20), colormap = (:viridis, 0.9))
    contour!(ax1, xgrid,tgrid, err_data, levels = range(lmin, lmax, length = 50), colormap = :plasma)
    ax1.xticks = [xgrid[1],L/2,xgrid[end]];
    ax1.yticks = [0, Tf/2, Tf];
        Colorbar(f[1, 2], co)
        display(f)    
    
    save(base_path*"error_abs_1d_Nx_40.pdf", f, pt_per_unit = 1)   
end










#=
begin
    fig = Figure(size=(800,600),fontsize=20)
    ax = Axis3(fig[1,1], azimuth = 3pi/4, 
                xlabel = "Position x in [m]", ylabel = "Time t in [s]", zlabel = "Error in [K]", 
                xlabelsize = 24, ylabelsize = 24, zlabelsize = 24,)

    surface!(ax, xgrid,tgrid[1:2:end], err_data[:,1:2:end], colormap = :plasma)
    lines!(ax, xgrid, [tgrid[end]], err_data[:,end]; linewidth = 4, color=:maroon)   
    xm, ym, zm = minimum(ax.finallimits[])
    zmin, zmax = minimum(err_data), maximum(err_data)
    #contour!(ax, xgrid,tgrid[1:2:end], err_data[:,1:2:end]; levels = 50, linewidth = 2, colorrange = (zmin, zmax), transformation = (:xy, zmin), transparency = true)       
    fig
    save("results/figures/"*"simulation_uncontrolled/compare_analytical_numerical_1d.pdf", fig,pt_per_unit = 1)    
end
=#