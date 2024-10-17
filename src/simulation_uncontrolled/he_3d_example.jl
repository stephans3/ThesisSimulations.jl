using Hestia 
λx = [10, 0.1];
λy = [10, 0.2];

property = DynamicAnisotropic(λx, λy, [8000], [400])
L, W = 0.3, 0.1;
Nx, Ny = 30, 10;
plate  = HeatPlate(L,W,Nx,Ny)
boundary = Boundary(plate)


# Emission / Stefan-Boltzmann Boundary Condition
# - heat transfer coefficient: k=10
# - heat radiation with emssivity ϵ=0.6
# - ambient temperature  Θamb=300 Kelvin
emission = Emission(10.0, 300.0, 0.6) 

# The emission is assumed on each boundary side of the 2D plate.
boundary = Boundary(plate)
setEmission!(boundary, emission, :west)
setEmission!(boundary, emission, :east)
setEmission!(boundary, emission, :north)

actuation = IOSetup(plate)
num_actuators = 3;
config = RadialCharacteristics(1, 2, 20);



setIOSetup!(actuation, plate, num_actuators, config ,:south)
### Simulation ###
function heat_conduction!(dw, w, param, t)
    u_in = 3e5*ones(num_actuators)*(1-cospi(2*t/Tf))
    diffusion!(dw, w, plate, property, boundary, actuation, u_in)
end

Tf = 180.0;
tspan = (0, Tf);
tsave = 60;
θinit = 300*ones(Nx*Ny);

using OrdinaryDiffEq
alg = KenCarp5();
prob = ODEProblem(heat_conduction!,θinit,tspan)
sol = solve(prob,alg, saveat=tsave)

# using Plots
# contourf(reshape(sol(100), Nx, Ny)')

data0 = reshape(sol[2], Nx, Ny)
data1 = reshape(sol[3], Nx, Ny)
data2 = reshape(sol[4], Nx, Ny)
#data3 = reshape(sol[5], Nx, Ny)

xgrid = 0 : L/(Nx-1) : L # = L/(2Nx) : L/Nx : L # Position in x-direction
ygrid = 0 : W/(Ny-1) : W # = W/(2Ny) : W/Ny : W # Position in x-direction

xgrid = 100*xgrid;
ygrid = 100*ygrid;

using CairoMakie
base_path = "results/figures/controlled/"
begin
    cmap = :plasma;

    f = Figure(size=(1200,900),fontsize=40,backgroundcolor = :transparent)
    ax1 = Axis(f[1, 1], title="Time: 60 s")
    tightlimits!(ax1)
    hidexdecorations!(ax1)
    contourf!(ax1, xgrid, ygrid, data0, levels = range(300, 550, length = 20), colormap=cmap)

    ax2 = Axis(f[2, 1], title="Time: 120 s")
    contourf!(ax2, xgrid, ygrid, data1,  levels = range(300, 550, length = 20), colormap=cmap)
    hidexdecorations!(ax2)
    tightlimits!(ax2)

    ax3 = Axis(f[3, 1], title="Time: 180 s")
#    hidexdecorations!(ax3)
    tightlimits!(ax3)

    co = contourf!(ax3, xgrid, ygrid, data2, levels = range(300, 550, length = 20), colormap=cmap)

    Colorbar(f[1:3, 2], co)
    f    
    save(base_path*"dynamic_anisotropic_2d.png", f, pt_per_unit = 1)
end

begin
    f = Figure(size=(1300,400),fontsize=20)
    ax1 = Axis(f[1, 1], title="Initial")
    tightlimits!(ax1)
    hidedecorations!(ax1)
    #contourf!(ax1, xgrid, ygrid, data0, levels = range(300.0, 470, length = 20))
    contourf!(ax1, xgrid, ygrid, data0)

 #   ax1.xticks = [0, L/2, L];
 #   ax1.yticks = [0, W/2, W];

    ax2 = Axis(f[2, 1], title="Proceeding")
    hideydecorations!(ax2)
    tightlimits!(ax2)
    #ax2.xticks = [L/2, L];
    # contourf!(ax2, xgrid, ygrid, data1, levels = range(0.0, 1.01, length = 20))
    contourf!(ax2, xgrid, ygrid, data1)

    ax3 = Axis(f[3, 1], title="Final")
    hideydecorations!(ax3)
    tightlimits!(ax3)
    ax3.xticks = [L/2, L];
#    co = contourf!(ax3, xgrid, ygrid, data2, levels = range(0.0, 1.01, length = 20))
    co = contourf!(ax3, xgrid, ygrid, data2)

    Colorbar(f[1:3, 2], co)
    f    

    # save(base_path*"anisotropic_heat_conduction.pdf", f, pt_per_unit = 1)   
end
