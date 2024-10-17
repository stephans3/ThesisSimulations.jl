using Hestia 
property = StaticAnisotropic([5e-6, 2e-5], 1, 1)

L = 0.1
W = 0.1
Nx = 20;
Ny = 20;
heatplate  = HeatPlate(L,W, Nx,Ny)
boundary = Boundary(heatplate)

### Simulation ###
function heat_conduction!(dw, w, param, t)
    diffusion!(dw, w, heatplate, property, boundary)
end

θinit = zeros(Nx,Ny)
@. θinit[7:Nx-6,7:Ny-6] = 1
θinit = reshape(θinit, Nx*Ny,1)

using OrdinaryDiffEq
tspan = (0, 10.0)
alg = KenCarp5()
prob= ODEProblem(heat_conduction!, θinit,tspan)
sol = solve(prob,alg, saveat=1.0)

xgrid = 0 : L/(Nx-1) : L # = L/(2Nx) : L/Nx : L # Position in x-direction
ygrid = 0 : W/(Ny-1) : W # = W/(2Ny) : W/Ny : W # Position in x-direction

base_path = "results/figures/modeling/"


using CairoMakie
begin
    data0 = reshape(sol[1], Nx, Ny)
    data1 = reshape(sol[6], Nx, Ny)
    data2 = reshape(sol[11], Nx, Ny)
    
    f = Figure(size=(1300,400),fontsize=26)
    ax1 = Axis(f[1, 1], title="Initial", xlabelsize = 30, ylabelsize = 30)
    tightlimits!(ax1)
    #hidedecorations!(ax1)
    contourf!(ax1, xgrid, ygrid, data0, levels = range(0.0, 1.01, length = 3))

    ax1.xticks = [0, L/2, L];
    ax1.yticks = [0, W/2, W];

    ax2 = Axis(f[1, 2], title="Proceeding", xlabelsize = 30, ylabelsize = 30)
    hideydecorations!(ax2)
    tightlimits!(ax2)
    ax2.xticks = [L/2, L];
    contourf!(ax2, xgrid, ygrid, data1, levels = range(0.0, 1.01, length = 20))

    ax3 = Axis(f[1, 3], title="Final", xlabelsize = 30, ylabelsize = 30)
    hideydecorations!(ax3)
    tightlimits!(ax3)
    ax3.xticks = [L/2, L];

    co = contourf!(ax3, xgrid, ygrid, data2, levels = range(0.0, 1.01, length = 20))
    Colorbar(f[1, 4], co)
    f    

    save(base_path*"anisotropic_heat_conduction.pdf", f, pt_per_unit = 1)   
end

