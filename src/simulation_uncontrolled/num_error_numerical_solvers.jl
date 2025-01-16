

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
λ = 50; # 45.0    # Thermal conductivity: constant
ρ = 8000; # 7800.0  # Mass density: constant
c = 400;# 480.0   # Specific heat capacity: constan
α = λ/(ρ*c) # Diffusivity

L = 0.2  # Length
p = 3e4 # 1e5;
temp_init(x) = p*x*(L-x);

Tf = 600;
Δt = 10;
t_samp = Δt;
tspan = (0.0, Tf)
tgrid = 0 : t_samp : Tf;
Nt = length(tgrid)

function heat_conduction_rod!(dθ, θ, param, t,rod1d)
    diffusion!(dθ, θ, rod1d, property, Boundary(rod1d))
end

using Hestia, OrdinaryDiffEq
property = StaticIsotropic(λ, ρ, c)
alg1 = ImplicitEuler()
alg2 = Trapezoid()
alg3 = KenCarp5()


Nx_orig = 100;
Δx = L/Nx_orig
xgrid_orig = L/(2Nx_orig) : Δx : L # Position in x-direction

sol_analytical = zeros(length(xgrid_orig), length(tgrid))
for (i,td) in enumerate(tgrid)
    for (k,xd) in enumerate(xgrid_orig)
        sol_analytical[k,i] = θ₀ + temp_evol(td,xd)
    end
end

θinit = θ₀ .+ temp_init.(xgrid_orig)
rod  = HeatRod(L, Nx_orig) # Numerical
heat_conduction!(dx,x,p,t) = heat_conduction_rod!(dx, x, p, t,rod)
prob = ODEProblem(heat_conduction!,θinit,tspan)

sol1_orig = solve(prob,alg1, dt=Δt ,saveat=t_samp)
sol2_orig = solve(prob,alg2, dt=Δt,saveat=t_samp)
sol3_orig = solve(prob,alg3, dt=Δt,saveat=t_samp)



using CairoMakie
base_path = "results/figures/simulation/"

100*xgrid_orig[1]

begin

    data1 = sol_analytical[1,:]
    data2 = sol_analytical[26,:]
    data3 = sol_analytical[51,:]

    fig1 = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(fig1[1, 1], xlabel = "Time t in [s]", ylabel = "Temperature in [K]", 
                xlabelsize = 30, ylabelsize = 30)#,limits = (nothing, (-2.7, 3.5))) 


    #ax1.xticks = Nx_arr;
    #ax1.yticks = -3: 1 : 3;

    lines!(ax1, tgrid, data1; linestyle = :dot, linewidth = 5, label = L"$x\approx 0$")
    lines!(ax1, tgrid, data2; linestyle = :dash, linewidth = 5, label = L"$x\approx L/4$")
    lines!(ax1, tgrid, data3; linestyle = :dashdot, linewidth = 5, label = L"$x\approx L/2$")

    
    axislegend(; position = :rb, backgroundcolor = (:grey90, 0.1));
    fig1
    save(base_path*"num_error_analytical_sol.pdf", fig1, pt_per_unit = 1)    
end


begin
    data0 = sol_analytical[1,:]
    data1 = Array(sol1_orig[1,:])
    data2 = Array(sol2_orig[1,:])
    data3 = Array(sol3_orig[1,:])

    fig1 = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(fig1[1, 1], xlabel = "Time t in [s]", ylabel = "Temperature in [K]", 
                xlabelsize = 30, ylabelsize = 30)#,limits = (nothing, (-2.7, 3.5))) 


    #ax1.xticks = Nx_arr;
    #ax1.yticks = -3: 1 : 3;

    tpoints = vcat(1,10:10:60)
    #lines!(ax1, tgrid, data1; linestyle = :solid, linewidth = 5, label = "Analytical", color=color = Makie.wong_colors()[4])
    lines!(ax1, tgrid, data1; linestyle = :dot, linewidth = 5, label = "Backward Euler")
    lines!(ax1, tgrid, data2; linestyle = :dash, linewidth = 5, label = "Trapezoidal Rule")
    lines!(ax1, tgrid, data3; linestyle = :dashdot, linewidth = 5, label = "KenCarp5")
  scatter!(ax1, tgrid[tpoints], data0[tpoints]; marker=:diamond, markersize=20, label = "Analytical Solution", color = :black) #Makie.wong_colors()[6])
  
    
    axislegend(; position = :rb, backgroundcolor = (:grey90, 0.1));
    fig1
    save(base_path*"num_error_temp_evolution_left.pdf", fig1, pt_per_unit = 1)    
end



Nx_arr = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
err_final_store = zeros(length(Nx_arr),3);
err_time_1_store = zeros(Nt, length(Nx_arr));
err_time_2_store = zeros(Nt, length(Nx_arr));
err_time_3_store = zeros(Nt, length(Nx_arr));

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

    sol1 = solve(prob,alg1, dt=Δt ,saveat=t_samp)
    sol2 = solve(prob,alg2, dt=Δt,saveat=t_samp)
    sol3 = solve(prob,alg3, dt=Δt,saveat=t_samp)

    err1 = temp_analytical - Array(sol1)
    err2 = temp_analytical - Array(sol2)
    err3 = temp_analytical - Array(sol3)

    err_time_1 = sum(abs2, err1,dims=1)*Δx
    err_time_2 = sum(abs2, err2,dims=1)*Δx
    err_time_3 = sum(abs2, err3,dims=1)*Δx
    
    err_time_1_store[:,jn] = err_time_1' 
    err_time_2_store[:,jn] = err_time_2' 
    err_time_3_store[:,jn] = err_time_3' 

    err_final_store[jn,1] = sum(abs2, err1)*t_samp*Δx
    err_final_store[jn,2] = sum(abs2, err2)*t_samp*Δx
    err_final_store[jn,3] = sum(abs2, err3)*t_samp*Δx 
end

err_time_1_data = mapreduce(i-> err_time_1_store[:,i],hcat,[1,5,10])
err_time_2_data = mapreduce(i-> err_time_2_store[:,i],hcat,[1,5,10])
err_time_3_data = mapreduce(i-> err_time_3_store[:,i],hcat,[1,5,10])



base_path = "results/figures/simulation/"
using CairoMakie


begin
    data1 = log10.(err_final_store[:,1])
    data2 = log10.(err_final_store[:,2])
    data3 = log10.(err_final_store[:,3])

    fig1 = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(fig1[1, 1], xlabel = "Number of Discretization Nodes", ylabel = L"Error Sum $\log_{10}(e_{\Sigma})$", 
                xlabelsize = 30, ylabelsize = 30)#,limits = (nothing, (-2.7, 3.5))) 


    ax1.xticks = Nx_arr;
    #ax1.yticks = -3: 1 : 3;
  
    scatterlines!(ax1, Nx_arr, data1; linestyle = :dot, marker=:circle, markersize=20, linewidth = 5, label = "Backward Euler")
    scatterlines!(ax1, Nx_arr, data2; linestyle = :dash,  marker=:circle, markersize=20, linewidth = 5, label = "Trapezoidal Rule")
    scatterlines!(ax1, Nx_arr, data3; linestyle = :solid,   marker=:circle, markersize=20, linewidth = 5, label = "KenCarp5")

    
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1));
    fig1
    save(base_path*"num_error_per_node.pdf", fig1, pt_per_unit = 1)    
end


begin
    data1 = log10.(err_time_1_data[:,1])
    data2 = log10.(err_time_1_data[:,2])
    data3 = log10.(err_time_1_data[:,3])

    fig1 = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(fig1[1, 1], xlabel = "Time t in [s]", ylabel = L"Error $\log_{10}(e)$", 
                xlabelsize = 30, ylabelsize = 30 )#,limits = (nothing, (-2.7, 3.5))) 


    #ax1.xticks = Nx_arr;
    #ax1.yticks = -3: 1 : 3;
  
    lines!(ax1, tgrid, data1; linestyle = :dot,  linewidth = 5, label = "N=10")
    lines!(ax1, tgrid, data2; linestyle = :dash, linewidth = 5, label = "N=50")
    lines!(ax1, tgrid, data3; linestyle = :solid,linewidth = 5, label = "N=100")

    
    axislegend(; position = :rb, backgroundcolor = (:grey90, 0.1));
    fig1
    save(base_path*"num_error_time_euler.pdf", fig1, pt_per_unit = 1)    
end


begin
    data1 = log10.(err_time_2_data[:,1])
    data2 = log10.(err_time_2_data[:,2])
    data3 = log10.(err_time_2_data[:,3])

    fig1 = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(fig1[1, 1], xlabel = "Time t in [s]", ylabel = L"Error $\log_{10}(e)$", 
                xlabelsize = 30, ylabelsize = 30,limits = (nothing, (-8,2))) 


    #ax1.xticks = Nx_arr;
    #ax1.yticks = -3: 1 : 3;
  
    lines!(ax1, tgrid, data1; linestyle = :dot,  linewidth = 5, label = "N=10")
    lines!(ax1, tgrid, data2; linestyle = :dash, linewidth = 5, label = "N=50")
    lines!(ax1, tgrid, data3; linestyle = :solid,linewidth = 5, label = "N=100")

    
    axislegend(; position = :rb, backgroundcolor = (:grey90, 0.1));
    fig1
    save(base_path*"num_error_time_trapezoidal.pdf", fig1, pt_per_unit = 1)    
end

begin
    data1 = log10.(err_time_3_data[:,1])
    data2 = log10.(err_time_3_data[:,2])
    data3 = log10.(err_time_3_data[:,3])

    fig1 = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(fig1[1, 1], xlabel = "Time t in [s]", ylabel = L"Error $\log_{10}(e)$", 
                xlabelsize = 30, ylabelsize = 30, limits = (nothing, (-8,0))) 


    #ax1.xticks = Nx_arr;
    #ax1.yticks = -3: 1 : 3;
  
    lines!(ax1, tgrid, data1; linestyle = :dot,  linewidth = 5, label = "N=10")
    lines!(ax1, tgrid, data2; linestyle = :dash, linewidth = 5, label = "N=50")
    lines!(ax1, tgrid, data3; linestyle = :solid,linewidth = 5, label = "N=100")

    
    axislegend(; position = :rb, backgroundcolor = (:grey90, 0.1));
    fig1
    save(base_path*"num_error_time_kencarp.pdf", fig1, pt_per_unit = 1)    
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

#=
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
=#









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