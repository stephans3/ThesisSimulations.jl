


λ = 50;
ρ = 8000; # Density
c = 400;  # Specific heat capacity

L = 0.1; # Length of 1D rod
W = 0.05
N₁ = 20;
N₂ = 20;
Nc = N₁*N₂ 
Δx₁ = L/N₁
Δx₂ = W/N₂

using Hestia 

property = StaticIsotropic(λ, ρ, c)
plate  = HeatPlate(L,W, N₁,N₂)
boundary = Boundary(plate)


### Actuation ###
actuation_wide = IOSetup(plate)
actuation_small = IOSetup(plate)
num_actuators = 1        # Number of actuators per boundary

config_wide  = RadialCharacteristics(1.0, 2, 30)    
config_small  = RadialCharacteristics(1.0, 2, 100)

setIOSetup!(actuation_wide, plate, num_actuators, config_wide, :south)
setIOSetup!(actuation_small, plate, num_actuators, config_small, :south)


function input(t)
    return 2e5;  #exp(12.1)
end


function heat_conduction!(dθ, θ, param, t)
    u1 = input(t)
    u_in = [u1]
    actuation = param
    diffusion!(dθ, θ, plate, property, boundary, actuation, u_in)
end



Tf = 120;
tspan = (0.0, Tf)
θ₀ = 300;
Ntotal = N₁*N₂
θinit = θ₀*ones(Ntotal)
dΘ = similar(θinit)

heat_conduction!(dΘ,θinit,actuation_wide,0)

# Δt = 1e-2               # Sampling time

using OrdinaryDiffEq

t_samp = 1.0;#1.0
alg = KenCarp5()
prob = ODEProblem(heat_conduction!,θinit,tspan)
sol_wide = solve(prob,alg,p=actuation_wide, saveat=t_samp)
sol_small = solve(prob,alg,p=actuation_small, saveat=t_samp)

maximum(Array(sol_wide))
maximum(Array(sol_small))

reshape(Array(sol_small[end]),N₁,N₂)
reshape(Array(sol_wide[end]),N₁,N₂)

base_path = "results/figures/controlled/"
# path2plot = base_path*"actuator_char_"*string(mode_char)*".pdf"

using CairoMakie
begin

    data0 = reshape(sol_small[end], N₁, N₂)
    xgrid = 0 : L/(N₁-1) : L
    ygrid = 0 : W/(N₂-1) : W
    mm_scale = 100;
    xgrid = mm_scale*xgrid
    ygrid = mm_scale*ygrid

    f = Figure(size=(518,400),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel="Length in [mm]", ylabel="Width in [mm]", xlabelsize = 30, ylabelsize = 30)
    tightlimits!(ax1)
    #hidedecorations!(ax1)
    co = contourf!(ax1, xgrid, ygrid, data0, colormap=:plasma, levels = range(300, 460, length = 50))
    contour!(ax1, xgrid, ygrid, data0, colormap=:plasma, levels = range(300, 460, length = 50))
    #Colorbar(f[1, 2], co)
    f    
    #save(base_path*"actuator_char_small.pdf", f, pt_per_unit = 1)   
end

begin

    data0 = reshape(sol_wide[end], N₁, N₂)
    xgrid = 0 : L/(N₁-1) : L
    ygrid = 0 : W/(N₂-1) : W
    mm_scale = 100;
    xgrid = mm_scale*xgrid
    ygrid = mm_scale*ygrid

    f = Figure(size=(600,400),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel="Length in [mm]", ylabel="Width in [mm]", xlabelsize = 30, ylabelsize = 30)
    tightlimits!(ax1)
    #hidedecorations!(ax1)
    co = contourf!(ax1, xgrid, ygrid, data0, colormap=:plasma, levels = range(300, 460, length = 50))
    contour!(ax1, xgrid, ygrid, data0, colormap=:plasma, levels = range(300, 460, length = 50))
    Colorbar(f[1, 2], co)
    f    
    save(base_path*"actuator_char_wide.pdf", f, pt_per_unit = 1)   
end


begin
    data0 = reshape(sol[61], N₁, N₂)
    data1 = reshape(sol[181], N₁, N₂)
    data2 = reshape(sol[301], N₁, N₂)

    data_max = ceil(maximum(data2)/10)*10

    xgrid = 0 : L/(N₁-1) : L
    ygrid = 0 : W/(N₂-1) : W
    mm_scale = 100;
    xgrid = mm_scale*xgrid
    ygrid = mm_scale*ygrid


    f = Figure(size=(1300,400),fontsize=26)

    # Label(f[2, 1], "60 seconds")
    # Label(f[2, 2], "180 seconds")
    # Label(f[2, 3], "300 seconds")

    ax1 = Axis(f[1, 1], title="60 seconds", xlabel="Length in [mm]", ylabel="Width in [mm]", xlabelsize = 30, ylabelsize = 30)
    tightlimits!(ax1)
    #hidedecorations!(ax1)
    contourf!(ax1, xgrid, ygrid, data0, colormap=:plasma, levels = range(300, data_max, length = 20))

    ax1.xticks = mm_scale*[0, L/2, L];
    ax1.yticks = mm_scale*[0, W/2, W];

    ax2 = Axis(f[1, 2], title="180 seconds", xlabel="Length in [mm]", xlabelsize = 30, ylabelsize = 30)
    hideydecorations!(ax2)
    tightlimits!(ax2)
    ax2.xticks = mm_scale*[L/2, L];
    contourf!(ax2, xgrid, ygrid, data1,colormap=:plasma,  levels = range(300,  data_max, length = 20))

    ax3 = Axis(f[1, 3], title="300 seconds", xlabel="Length in [mm]", xlabelsize = 30, ylabelsize = 30)
    hideydecorations!(ax3)
    tightlimits!(ax3)
    ax3.xticks = mm_scale*[L/2, L];

    co = contourf!(ax3, xgrid, ygrid, data2, colormap=:plasma, levels = range(300,  data_max, length = 20))
    
    if mode_char == 2
        Colorbar(f[1, 4], co, ticks = [300, 340, 380, 420])
    elseif mode_char == 3
        Colorbar(f[1, 4], co, ticks = [300, 350, 400, 450, 500])
    else
        Colorbar(f[1, 4], co)
    end
    f    

    save(path2plot, f, pt_per_unit = 1)   
end













using Plots
plot(sol, legend=false)
n_samp = length(sol.t)
data1 = reshape(Array(sol),n_samp,N₂,N₁)


contourf(reshape(Array(sol)[:,51], N₂,N₁)')
plot()

contourf(data1[101,:,:])
