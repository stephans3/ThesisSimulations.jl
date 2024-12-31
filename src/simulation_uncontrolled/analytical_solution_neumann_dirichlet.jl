
function neumann_temp_evol(t,x;kmax=100)
    c0 = p*L^2 / 6
    c1 = p*L^2 / pi^2

    s = 0;

    for k=1 : kmax
        s += exp(-4*α*(k*π/L)^2*t) *cospi(2*k*x/L) / k^2;
    end

    return c0 - c1*s
end

function dirichlet_temp_evol(t,x;kmax=100)
    c1 = 8p*L^2 / pi^3
    s = 0;
    for k=1 : kmax
        s += exp(-α*((2k-1)*π/L)^2*t) *sinpi((2*k-1)*x/L) / (2k-1)^3;
    end
    return c1*s
end

α = 0.1;
L = 1;
Nx = 41;
p = 4/L^2;
xgrid = 0 : L/(Nx-1) : L # Position in x-direction
Tf = 1;
t_samp = 0.25;
tgrid = 0 : t_samp : Tf;

temp_neumann = zeros(length(xgrid), length(tgrid))
temp_dirichlet = zeros(length(xgrid), length(tgrid))
for (i,td) in enumerate(tgrid)
    for (k,xd) in enumerate(xgrid)
        temp_neumann[k,i] = neumann_temp_evol(td,xd)
        temp_dirichlet[k,i] = dirichlet_temp_evol(td,xd)
    end
end

temp_neumann
temp_dirichlet




#=
using Plots
plot(temp_neumann)
plot(temp_dirichlet)
surface(temp_neumann)
surface(temp_dirichlet)
plot(xgrid, temp_dirichlet[:,2])
=#



using CairoMakie
path2folder = "results/figures/simulation/"
begin
    filename = path2folder*"neumann_1.pdf"

    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Position x", ylabel = "Temperature", xlabelsize = 30, ylabelsize = 30, limits = (nothing, (-0.1, 1.5)),)
 
    ax1.yticks = [0, 0.5, 1, 1.5];
    lines!(xgrid, temp_neumann[:,1], linestyle = :solid,  linewidth = 5, label=L"$t=0$")
    lines!(xgrid, temp_neumann[:,2], linestyle = :dash,  linewidth = 5, label=L"$t=0.25$")
    lines!(xgrid, temp_neumann[:,5], linestyle = :dot,  linewidth = 5, label=L"$t=1$")

    axislegend(; position = :rt, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f

    #save(filename, f, pt_per_unit = 1)   
end

begin
    filename = path2folder*"dirichlet_1.pdf"

    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Position x", ylabel = "Temperature", xlabelsize = 30, ylabelsize = 30, limits = (nothing, (-0.1, 1.5)),)
 

    ax1.yticks = [0, 0.5, 1, 1.5];
    lines!(xgrid, temp_dirichlet[:,1], linestyle = :solid,  linewidth = 5, label=L"$t=0$")
    lines!(xgrid, temp_dirichlet[:,2], linestyle = :dash,  linewidth = 5, label=L"$t=0.25$")
    lines!(xgrid, temp_dirichlet[:,5], linestyle = :dot,  linewidth = 5, label=L"$t=1$")

    axislegend(; position = :rt, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f
    
    #save(filename, f, pt_per_unit = 1)   
end


begin
    filename = path2folder*"neumann_contour.pdf"
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Position x", ylabel = "Time t", xlabelsize = 30, ylabelsize = 30)#,limits = ((-0.01, 1.01), nothing ))
    tightlimits!(ax1)
    # hidedecorations!(ax1)
    #x1grid_cm = 100*x1grid
    #x2grid_cm = 100*x2grid
    co = contourf!(ax1, xgrid,tgrid, temp_neumann, colormap=:plasma, levels = range(0.0, 1.0, length = 20))
    ax1.xticks = 0 : 0.25 : 1;
    # ax1.yticks = 0 : 1 : 5;
    Colorbar(f[1, 2], co)
    f    

    #save(filename, f, pt_per_unit = 1)   
end


begin
    filename = path2folder*"dirichlet_contour.pdf"
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Position x", ylabel = "Time t", xlabelsize = 30, ylabelsize = 30)#,limits = ((-0.01, 1.01), nothing ))
    tightlimits!(ax1)
    # hidedecorations!(ax1)
    #x1grid_cm = 100*x1grid
    #x2grid_cm = 100*x2grid
    co = contourf!(ax1, xgrid,tgrid, temp_dirichlet, colormap=:plasma, levels = range(0.0, 1.0, length = 20))
    ax1.xticks = 0 : 0.25 : 1;
    # ax1.yticks = 0 : 1 : 5;
    Colorbar(f[1, 2], co)
    f    

    #save(filename, f, pt_per_unit = 1)   
end


begin
    filename = path2folder*"dirichlet_2.pdf"
    f = Figure(size=(600,400),fontsize=26)
 
    ax = Axis3(f[1,1], azimuth = 7pi/4, 
                xlabel = "Position x", ylabel = "Time t", zlabel = "Temperature", 
                xlabelsize = 30, ylabelsize = 30, zlabelsize = 30,)

    surface!(ax, xgrid,tgrid, temp_dirichlet, colormap = :plasma)     
    f
    save(filename, f, pt_per_unit = 1)   
end
