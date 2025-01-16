

L = 1# 0.1; # Length of 1D rod
W = 1# 0.1
N₁ = 20;
N₂ = 20;
Nc = N₁*N₂ 
Δx₁ = L/N₁
Δx₂ = W/N₂


xgrid = 0 : L/(N₁-1) : L
ygrid = 0 : W/(N₂-1) : W

x0 = L/2
y0 = W/2

using LinearAlgebra
# M = 30diagm(ones(2))
# M = 30diagm([1,0.5])
M = 30*[1 0;1 0]

det(M)

ν = 10
char(x,y;p=2) = exp(-norm(M*([x,y] - [x0,y0]),p)^(ν))


xgrid = 0 : L/(N₁-1) : L
ygrid = 0 : W/(N₂-1) : W
M = diagm([4,4])

data0 = char.(xgrid', ygrid,p=1)
data1 = char.(xgrid', ygrid,p=Inf)

using Plots
surface(xgrid,ygrid,data0)
surface(xgrid,ygrid,data1)










# using Plots
# contourf(data)


base_path = "results/figures/controlled/"
path2plot = base_path*"characteristics_io_2d.pdf"
using CairoMakie
begin
    xgrid = 0 : L/(N₁-1) : L
    ygrid = 0 : W/(N₂-1) : W
    M = 4diagm(ones(2))
    data0 = char.(xgrid', ygrid)

    M = 4diagm([2,0.5])
    data1 = char.(xgrid', ygrid)

    M = 4*[1 0;1 0]
    data2 = char.(xgrid', ygrid)

    M = 4*[1 1/2;1/2 1]
    data3 = char.(xgrid', ygrid)


    mm_scale = 100;
    xgrid = mm_scale*xgrid
    ygrid = mm_scale*ygrid


    f = Figure(size=(1300,400),fontsize=26)

    # Label(f[2, 1], "60 seconds")
    # Label(f[2, 2], "180 seconds")
    # Label(f[2, 3], "300 seconds")

    ax1 = Axis(f[1, 1], xlabel="Length in [mm]\n (a) Circle", ylabel="Width in [mm]", xlabelsize = 30, ylabelsize = 30)
    tightlimits!(ax1)
    #hidedecorations!(ax1)
    contourf!(ax1, xgrid, ygrid, data0', colormap=:plasma, levels = range(0, 1, length = 20))

    ax1.xticks = mm_scale*[0, L/2, L];
    ax1.yticks = mm_scale*[0, W/2, W];

    #=
    ax1b = Axis(f[2, 1], xlabel="Length in [mm]\n (a) Circle", xlabelsize = 30, ylabelsize = 30)
    # tightlimits!(ax1)
    #hidedecorations!(ax1)
    lines!(ax1b, xgrid, data0[10,:])
    ax1b.xticks = mm_scale*[0, L/2, L];
    #ax1.yticks = mm_scale*[0, W/2, W];
    =#


    ax2 = Axis(f[1, 2], xlabel="Length in [mm]\n (b) Ellipse", xlabelsize = 30, ylabelsize = 30)
    hideydecorations!(ax2)
    tightlimits!(ax2)
    ax2.xticks = mm_scale*[L/2, L];
    contourf!(ax2, xgrid, ygrid, data1',colormap=:plasma,  levels = range(0, 1, length = 20))

    #=
    ax2b = Axis(f[2, 2], xlabel="Length in [mm]\n (a) Circle", xlabelsize = 30, ylabelsize = 30)
    # tightlimits!(ax1)
    #hidedecorations!(ax1)
    lines!(ax2b, xgrid, data1[10,:])

    ax2b.xticks = mm_scale*[0, L/2, L];
    #ax1.yticks = mm_scale*[0, W/2, W];
    =#


    ax3 = Axis(f[1, 3], xlabel="Length in [mm]\n (c) Rect", xlabelsize = 30, ylabelsize = 30)
    hideydecorations!(ax3)
    tightlimits!(ax3)
    ax3.xticks = mm_scale*[L/2, L];
    contourf!(ax3, xgrid, ygrid, data2',colormap=:plasma,  levels = range(0, 1, length = 20))

    ax4 = Axis(f[1, 4], xlabel="Length in [mm]\n (d) Rotation", xlabelsize = 30, ylabelsize = 30)
    hideydecorations!(ax4)
    tightlimits!(ax4)
    ax4.xticks = mm_scale*[L/2, L];
    co = contourf!(ax4, xgrid, ygrid, data3', colormap=:plasma, levels = range(0, 1, length = 20))
    Colorbar(f[1, 5], co)
    
    f    

    # save(path2plot, f, pt_per_unit = 1)   
end

