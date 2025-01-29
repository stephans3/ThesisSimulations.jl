
# Compute thermal conductivity
# x₁ direction

θvec = collect(300:50:500)
M_temp = mapreduce(z-> [1  z  z^2 z^3 z^4], vcat, θvec)

λ1data = [40,44,50,52,52.5]
λ1p = inv(M_temp)*λ1data

λ2data = [40,55,60,65,68]
λ2p = inv(M_temp)*λ2data

ρ = 8000; # Density
c = 400;  # Specific heat capacity

L = 0.3; # Length of 1D rod
W = 0.05
N₁ = 30;
N₂ = 10;
Nc = N₁*N₂ 
Δx₁ = L/N₁
Δx₂ = W/N₂

using Hestia 

# property = StaticAnisotropic(40, 40, ρ,c)

property = DynamicAnisotropic(λ1p, λ2p, [ρ],[c])

plate  = HeatPlate(L,W, N₁,N₂)
boundary = Boundary(plate)

### Boundaries ###
θamb = 300.0;
h = 10;
Θamb = 300;
ε = 0.1;
emission = Emission(h, Θamb,ε) #  Stefan-Boltzmann BC: heat transfer (linear) and heat radiation (quartic/nonlinear)
boundary = Boundary(plate)

setEmission!(boundary, emission, :west )
setEmission!(boundary, emission, :east )
setEmission!(boundary, emission,  :north )

### Actuation ###
actuation = IOSetup(plate)
num_actuators = 3        # Number of actuators per boundary
config  = RadialCharacteristics(1.0, 2, 30)
setIOSetup!(actuation, plate, num_actuators, config, :south)


function input_obc(t,p)
    return exp(p[1] - p[3]^2 * (t/Tf - 1/p[2])^2)
end

Tf = 1200;
ts = (Tf/1000)
tgrid = 0 : ts : Tf

p_fbc = [11.81522609265776
            2.070393374741201
            9.212088025161702]

# Reference
ref_init = 300;
Δr = 200; # Difference operating points
ps = 10; # Steepness
ψ(t) = (1+tanh(10*(t/Tf - 1/2)))/2
ref(t) = ref_init + Δr*ψ(t)


# Internal Energy
ΔU = ρ *c *L*W*Δr

# Emitted thermal energy
E_tr = (h*(ts*sum(ref.(tgrid[2:end])) - Tf*Θamb))
coeff_rad = ε*5.67*1e-8;
E_rad = (coeff_rad*ts*sum(ref.(tgrid[2:end]).^4))
E_em_approx =  (L+2W)*(E_tr + E_rad)

using SpecialFunctions
u_in_energy(p₁,p₂,p₃) = exp(p₁)*Tf*sqrt(pi) * (erf(p₃-p₃/p₂) - erf(-p₃/p₂)) / (2 * p₃)
E_oc_fbc_approx = u_in_energy(p_fbc...)

act_char_int = (sum(actuation.character[:south])*Δx₁)
E_in_fbc_approx = E_oc_fbc_approx*act_char_int

E_needed = ΔU +E_em_approx
E_necessary_perc = 100*E_in_fbc_approx / E_needed


function loss_energy(u,p)
    return (E_em_approx + ΔU - act_char_int*u_in_energy(u[1],p[1],u[2]))^2 / Tf
end


#loss_energy([p_fbc[1],p_fbc[3]],p₂)

const store_loss=[]
global store_param=[]

callback = function (state, l) 
    # store loss and parameters
    append!(store_loss,l) # Loss values 
    
    # store_param must be global
    global store_param  
    store_param = vcat(store_param,[state.u]) # save with vcat

    #println("iter")

    return false
end


# Find optimal p3
using Optimization, OptimizationOptimJL, ForwardDiff
opt_p = [p_fbc[2]]
opt_u0 = [p_fbc[1],p_fbc[3]]
loss_energy(opt_u0, opt_p)

optf = OptimizationFunction(loss_energy, Optimization.AutoForwardDiff())
opt_prob = OptimizationProblem(optf, opt_u0, opt_p)
p13 = Optimization.solve(opt_prob, ConjugateGradient(), callback=callback, maxiters=100)






p₂ = p_fbc[2]
p1grid = 11.7 : 0.05 : 12.9
p3grid = 9.09 : 0.01 : 9.3

loss_data = zeros(length(p1grid),length(p3grid))

for (i1,p1) in enumerate(p1grid), (i3,p3) in enumerate(p3grid)
    loss_data[i1,i3] = loss_energy([p1,p3],p₂)
end

p13_path = hcat(store_param...)'

using CairoMakie
path2folder = "results/figures/controlled/"


begin
    filename = path2folder*"feedforward_energy_log_loss.pdf"
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Iteration Number", ylabel = "Logarithmic Loss", xlabelsize = 30, ylabelsize = 30)
    num_iterations = 0:length(pars_data[:,1])-1;
    scatterlines!(num_iterations, log10.(store_loss), linestyle = :dash,  linewidth = 3, markersize=20)
    f
    save(filename, f, pt_per_unit = 1)   
end




begin
    filename = path2folder*"feedforward_energy_contour.pdf"
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = L"Gain $p_{1}$", ylabel = L"Kurtosis $p_{3}$", xlabelsize = 30, ylabelsize = 30)
    scale = 1e-10
    #tightlimits!(ax1)
    #hidedecorations!(ax1)
    co = contourf!(ax1, p1grid, p3grid, scale*loss_data, levels=20, colormap=:managua) #levels = range(0.0, 10.0, length = 20))
    lines!(p13_path[1:end,1],p13_path[1:end,2], linestyle = :dash,  linewidth = 5, color=:black)#RGBf(0.5, 0.2, 0.8))
    scatter!([p13[1]],[p13[2]], marker = :xcross, markersize=25, color=:purple)
    # ax1.xticks = 11.22 : 0.04 : 11.34 #[11.2, 11.24, 11.28, 11.32, 11.36];
    # ax1.yticks = 2.05:0.05:2.2;
    Colorbar(f[1, 2], co,label = L"Error $\times 10^{10}$")
    f    

    save(filename, f, pt_per_unit = 1)   
end


pars_data = hcat(store_param...)'

begin
    filename = path2folder*"feedforward_energy_p1_var.pdf"
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Iteration Number", ylabel = L"Gain $p_{1}$", xlabelsize = 30, ylabelsize = 30)

    num_iterations = 0:length(pars_data[:,1])-1;
    
    ax1.yticks = 11.8 : 0.2 : 12.4
    scatterlines!(num_iterations, pars_data[:,1], linestyle = :dash,  linewidth = 3, markersize=20)

    f
    save(filename, f, pt_per_unit = 1)   
end


begin
    filename = path2folder*"feedforward_energy_p3_var.pdf"
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Iteration Number",  ylabel = L"Kurtosis $p_{3}$", xlabelsize = 30, ylabelsize = 30)

    num_iterations = 0:length(pars_data[:,1])-1;
    
    ax1.yticks = 9.15 : 0.02 : 9.22 #[11.2, 11.24, 11.28, 11.32, 11.36];   
    scatterlines!(num_iterations, pars_data[:,2], linestyle = :dash,  linewidth = 3, marker = :xcross, markersize=20)

    f
    save(filename, f, pt_per_unit = 1)   
end







function heat_conduction!(dθ, θ, param, t)

    u1 = input_obc(t,param[1:3])
    u2 = input_obc(t,param[4:6])
    u3 = input_obc(t,param[7:9])
    
    u_in = [u1, u2, u3]
    diffusion!(dθ, θ, plate, property, boundary, actuation, u_in)
end



tspan = (0.0, Tf)
θ₀ = 300;
Ntotal = N₁*N₂
θinit = θ₀*ones(Ntotal)


p_energy = [p13[1],p₂,p13[2]] 
pinit = repeat(p_energy,3)

dΘ = similar(θinit)
heat_conduction!(dΘ,θinit,pinit,0)

# Δt = 1e-2               # Sampling time

using OrdinaryDiffEq
t_samp = 10.0
alg = KenCarp5()
prob = ODEProblem(heat_conduction!,θinit,tspan,pinit)
sol = solve(prob,alg,p=pinit, saveat=t_samp)

input1(t) = input_obc(t,p_energy[1:3])
input1_data = input1.(sol.t)

ref_data_2 = ref.(sol.t)


### Sensor ###
num_sensor = 3        # Number of sensors
sensing = IOSetup(plate)
config_sensor  = RadialCharacteristics(1.0, 1, 0)
setIOSetup!(sensing, plate, num_sensor, config_sensor, :north)

C = zeros(num_sensor,Ntotal)
b_sym = :north
ids = unique(sensing.identifier[b_sym])
for i in ids
    idx = findall(x-> x==i, sensing.identifier[b_sym])
    boundary_idx = sensing.indices[b_sym]
    boundary_char = sensing.character[b_sym]
    C[i,boundary_idx[idx]] = boundary_char[idx] / sum(boundary_char[idx])
end

y = C*Array(sol)


begin   
    filename = path2folder*"feedforward_energy_input.pdf"

    f = Figure(size=(600,400),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel = "Time t in [s]", ylabel=L"Input $\times 10^5$",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    scale = 1e-5;

    ax1.xticks = 0 : 200 : 1200;
    #ax1.yticks = 0 : 2.5 : 10;

    tstart = 450;
    tend = 2550;
    lines!(sol.t, scale*input1_data;   linestyle = :dash,   linewidth = 5, label="Input 1")
    # lines!(sol_opt.t, scale*input2_data;   linestyle = :dash,   linewidth = 5, label="Input 2")
    # axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);
    

    # ax2 = Axis(f, bbox=BBox(420, 570, 252, 370), ylabelsize = 24)
    # ax2.xticks = 800 : 200 : Tf;
    # ax2.yticks = [1.8, 2];
    # lines!(sol_opt.t[54:66], scale*input1_data[54:66];   linestyle = :dot,   linewidth = 5)
    # lines!(sol_opt.t[54:66], scale*input2_data[54:66];   linestyle = :dash,   linewidth = 5)
    # translate!(ax2.scene, 0, 0, 10);

    f  
    save(filename, f, pt_per_unit = 1)   
end



begin   
    filename = path2folder*"feedforward_energy_output.pdf"
    f = Figure(size=(600,400),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel = "Time t in [s]", ylabel="Temperature in [K]",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    # scale = 1e-4;

    ax1.xticks = 0 : 200 : 1200;
    # ax1.yticks = 300 : 25 : 400;

    tstart = 450;
    tend = 2550;
    lines!(sol.t, y[1,:];   linestyle = :dot,   linewidth = 5, label="Output 1")
    lines!(sol.t, y[2,:];   linestyle = :dash,   linewidth = 5, label="Output 2")
    scatter!(sol.t[1:15:end], ref_data_2[1:15:end];  markersize = 15, marker = :diamond, color=:black, label = "Reference")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);
     
    
    ax2 = Axis(f, bbox=BBox(410, 560, 140, 260), ylabelsize = 24)
    ax2.xticks = 900 : 150 : Tf;
    ax2.yticks = [500, 505];
    lines!(sol.t[91:end], y[1,91:end];   linestyle = :dot,   linewidth = 5)
    lines!(sol.t[91:end], y[2,91:end];   linestyle = :dash,   linewidth = 5)
    scatter!(sol.t[91:15:end], ref_data_2[91:15:end];  markersize = 15, marker = :diamond, color=:black, label = "Reference")
    translate!(ax2.scene, 0, 0, 10);
    elements = keys(ax2.elements)
    filtered = filter(ele -> ele != :xaxis && ele != :yaxis, elements)
    foreach(ele -> translate!(ax2.elements[ele], 0, 0, 9), filtered)
    
    f  
    save(filename, f, pt_per_unit = 1)   
end





