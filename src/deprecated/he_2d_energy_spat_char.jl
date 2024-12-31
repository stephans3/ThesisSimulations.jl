
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

actuation.character[:south]

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


act_sc_inv = 1/(sum(actuation.character[:south])*Δx₁) # Inverse of spatial characteristics of actuator
E_oc_mean = act_sc_inv * ΔU

u₀ = 1e-1;
p₂ = p_fbc[2]
function energy_em_approx!(F, x)
    F[1] = u₀- input_obc(0,[x[1], p₂, x[2]])
    F[2] = E_em_approx + E_oc_mean - u_in_energy(x[1], p₂, x[2])
end

using NLsolve
sol_nl_em_approx = nlsolve(energy_em_approx!, [p_fbc[1],p_fbc[2]])
p12,p32 = sol_nl_em_approx.zero

p_energy = [p12,p₂,p32] 

#=
p_energy = [12.198313924845424
             2.070393374741201
             7.884067490533799]
=#

u_test(t) = input_obc(t,p_energy)

# using Plots
# plot(tgrid, u_test.(tgrid))


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

dΘ = similar(θinit)

# pinit = repeat(p_fbc,3)
pinit = repeat(p_energy,3)

heat_conduction!(dΘ,θinit,pinit,0)

# Δt = 1e-2               # Sampling time

using OrdinaryDiffEq

t_samp = 10.0
alg = KenCarp5()
prob = ODEProblem(heat_conduction!,θinit,tspan,pinit)
sol = solve(prob,alg,p=pinit, saveat=t_samp)

length(sol.t)

# using Plots
# heatmap(reshape(Array(sol)[:,110],N₁,N₂)')

ref_data = repeat(ref.(sol.t)',3)

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
err = ref_data - y
sum(abs2, (sol.t/Tf)' .* err) /Tf

function loss_optim_std(u,p)
    pars = u
    sol_loss = solve(prob, alg,p=pars, saveat = t_samp)
    if sol_loss.retcode == ReturnCode.Success
        y = C*Array(sol_loss)
        err = ref_data - y
        loss = sum(abs2, err) / Tf
    else
        loss = Inf
    end
    return loss
end

const store_loss=[]
global store_param=[]

callback = function (state, l) 
    # store loss and parameters
    append!(store_loss,l) # Loss values 
    
    # store_param must be global
    global store_param  
    store_param = vcat(store_param,[state.u]) # save with vcat


    println(l)
    #println("iter")

    if l < -0.2
        return true
    end

    return false
end


p_opt = repeat(p_energy,3)
loss_optim_std(p_opt, [0])

using Optimization, OptimizationOptimJL, ForwardDiff
optf = OptimizationFunction(loss_optim_std, Optimization.AutoForwardDiff())
opt_prob = OptimizationProblem(optf, p_opt, [0])
p_final = Optimization.solve(opt_prob, ConjugateGradient(), callback=callback, maxiters=15)

#=
p_final = [
12.23289015172492
  2.056391984862441
  7.879402492310628
 12.214983965462167
  2.066088422337319
  7.882184268166304
 12.232890151724952
  2.056391984862413
  7.879402492310624
  ]
=#

#=
store_loss = [
4.9845834527496855
 1.5946995355057922
 1.2819854856936472
 1.1035727954160672
 1.0951496379003853
 1.0933765495019603
 1.0916258279325208
 1.0873712138294929
 1.0823621972036261
 1.0821437202383863
 1.0820275335084648
 1.0816877957004245
 1.081499714751307
 1.0812581907004104
 1.0812487196693439
 1.0812458915186807
 ]
=#

loss_optim_std(p_final, [0])

sol_final = solve(prob,alg,p=p_final, saveat=t_samp)

y_final = C*Array(sol_final)
st_pars[1,:]
st_pars[4,:]
[st_pars[1,:],st_pars[4,:]]
st_pars = hcat(store_param...)

store_param

using CairoMakie
path2folder = "results/figures/controlled/"
begin
    filename = path2folder*"feedforward_obc_parameter_1.pdf"
    pars_data = hcat(st_pars[1,:],st_pars[4,:])

    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Iteration Number", ylabel = L"Parameter $p_{1}$", xlabelsize = 30, ylabelsize = 30)
    
    num_iterations = 0:length(pars_data[:,1])-1;
    
    scatterlines!(num_iterations, pars_data[:,1], linestyle = :dash,  linewidth = 3, markersize=20, label="Actuator 1")
    scatterlines!(num_iterations, pars_data[:,2], linestyle = :dash,  linewidth = 3, marker = :xcross, markersize=20, label="Actuator 2")

    # lines!(num_iterations, pars_data[:,1], linestyle = :dash,  linewidth = 3)
    # lines!(num_iterations, pars_data[:,2], linestyle = :dash,  linewidth = 3)
    axislegend(; position = :rb, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f

    save(filename, f, pt_per_unit = 1)   
end



begin
    filename = path2folder*"feedforward_obc_parameter_2.pdf"
    pars_data = hcat(st_pars[2,:],st_pars[5,:])

    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Iteration Number", ylabel = L"Parameter $p_{2}$", xlabelsize = 30, ylabelsize = 30)
    
    num_iterations = 0:length(pars_data[:,1])-1;
    
    scatterlines!(num_iterations, pars_data[:,1], linestyle = :dash,  linewidth = 3, markersize=20, label="Actuator 1")
    scatterlines!(num_iterations, pars_data[:,2], linestyle = :dash,  linewidth = 3, marker = :xcross, markersize=20, label="Actuator 2")

    # lines!(num_iterations, pars_data[:,1], linestyle = :dash,  linewidth = 3)
    # lines!(num_iterations, pars_data[:,2], linestyle = :dash,  linewidth = 3)
    axislegend(; position = :rt, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f

    save(filename, f, pt_per_unit = 1)   
end

begin
    filename = path2folder*"feedforward_obc_parameter_3.pdf"
    pars_data = hcat(st_pars[3,:],st_pars[6,:])

    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Iteration Number", ylabel = L"Parameter $p_{3}$", xlabelsize = 30, ylabelsize = 30)
    
    num_iterations = 0:length(pars_data[:,1])-1;
    
    scatterlines!(num_iterations, pars_data[:,1], linestyle = :dash,  linewidth = 3, markersize=20, label="Actuator 1")
    scatterlines!(num_iterations, pars_data[:,2], linestyle = :dash,  linewidth = 3, marker = :xcross, markersize=20, label="Actuator 2")

    # lines!(num_iterations, pars_data[:,1], linestyle = :dash,  linewidth = 3)
    # lines!(num_iterations, pars_data[:,2], linestyle = :dash,  linewidth = 3)
    axislegend(; position = :rt, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f

    save(filename, f, pt_per_unit = 1)   
end

begin
    filename = path2folder*"feedforward_obc_loss.pdf"
    # pars_data = hcat(st_pars[3,:],st_pars[6,:])
    
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Iteration Number", ylabel = "Loss", xlabelsize = 30, ylabelsize = 30)
    
    num_iterations = 0:length(pars_data[:,1])-1;
    
    scatterlines!(num_iterations, store_loss, linestyle = :dash,  linewidth = 3, markersize=20)
    # scatterlines!(num_iterations, pars_data[:,2], linestyle = :dash,  linewidth = 3, marker = :xcross, markersize=20, label="Actuator 2")

    # lines!(num_iterations, pars_data[:,1], linestyle = :dash,  linewidth = 3)
    # lines!(num_iterations, pars_data[:,2], linestyle = :dash,  linewidth = 3)
    # axislegend(; position = :rt, backgroundcolor = (:grey90, 0.1), labelsize=30);

    ax2 = Axis(f, bbox=BBox(240, 575, 175, 360), ylabelsize = 24)
    ax2.xticks = 5 : 5 : 15;
    ax2.yticks = [1.085, 1.09];
    scatterlines!(ax2, num_iterations[6:end], store_loss[6:end], linestyle = :dash,  linewidth = 3, markersize=20)
    #lines!(ax2, num_iterations[5:end], store_loss[5:end];   linestyle = :dash,  linewidth = 5, color=Makie.wong_colors()[1])
    translate!(ax2.scene, 0, 0, 10);

    f

    save(filename, f, pt_per_unit = 1)   
end




input1(t) = input_obc(t,p_final[1:3])
input2(t) = input_obc(t,p_final[4:6])

input1_data = input1.(sol_final.t)
input2_data = input2.(sol_final.t)


begin   
    filename = path2folder*"feedforward_obc_input.pdf"

    f = Figure(size=(600,400),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel = "Time t in [s]", ylabel=L"Input $\times 10^5$",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    scale = 1e-5;

    ax1.xticks = 0 : 200 : 1200;
    #ax1.yticks = 0 : 2.5 : 10;

    tstart = 450;
    tend = 2550;
    lines!(sol_final.t, scale*input1_data;   linestyle = :dot,   linewidth = 5, label="Input 1")
    lines!(sol_final.t, scale*input2_data;   linestyle = :dash,   linewidth = 5, label="Input 2")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);
    

    ax2 = Axis(f, bbox=BBox(420, 570, 252, 370), ylabelsize = 24)
    #ax2.xticks = 800 : 200 : Tf;
    ax2.yticks = [1.8, 2];
    lines!(sol_final.t[54:66], scale*input1_data[54:66];   linestyle = :dot,   linewidth = 5)
    lines!(sol_final.t[54:66], scale*input2_data[54:66];   linestyle = :dash,   linewidth = 5)
    translate!(ax2.scene, 0, 0, 10);

    f  
    save(filename, f, pt_per_unit = 1)   
end

ref_data

begin   
    filename = path2folder*"feedforward_obc_output.pdf"
    f = Figure(size=(600,400),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel = "Time t in [s]", ylabel="Temperature in [K]",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    # scale = 1e-4;

    ax1.xticks = 0 : 200 : 1200;
    # ax1.yticks = 300 : 25 : 400;

    tstart = 450;
    tend = 2550;
    lines!(sol_final.t, y_final[1,:];   linestyle = :dot,   linewidth = 5, label="Output 1")
    lines!(sol_final.t, y_final[2,:];   linestyle = :dash,   linewidth = 5, label="Output 2")
    scatter!(sol_final.t[1:10:end], ref_data[1,1:10:end];  markersize = 15, marker = :diamond, color=:black, label = "Reference")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);
     
    
    ax2 = Axis(f, bbox=BBox(407, 570, 140, 260), ylabelsize = 24)
    ax2.xticks = 800 : 200 : Tf;
    ax2.yticks = [495, 500];
    lines!(sol_final.t[80:end], y_final[1,80:end];   linestyle = :dot,   linewidth = 5)
    lines!(sol_final.t[80:end], y_final[2,80:end];   linestyle = :dash,   linewidth = 5)
    scatter!(sol_final.t[81:20:end], ref_data[1,81:20:end];  markersize = 15, marker = :diamond, color=:black, label = "Reference")
    translate!(ax2.scene, 0, 0, 10);
    
    f  
    save(filename, f, pt_per_unit = 1)   
end

x1grid = Δx₁/2 : Δx₁ : L
x2grid = Δx₂/2 : Δx₂ : W

Array(sol_final)[(N₂-1)*N₁+1:N₂*N₁,1:20:end]
temp_data = Array(sol_final)

# filename = path2folder*"feedforward_temp_distribution_30.pdf"
begin
    data = reshape(temp_data[:,31],N₁,N₂)
    filename = path2folder*"feedforward_temp_distribution_30.pdf"
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = L"Length $x_{1}$ in [cm]", ylabel = L"Width $x_{2}$ in [cm]", xlabelsize = 30, ylabelsize = 30)
    # tightlimits!(ax1)
    # hidedecorations!(ax1)
    x1grid_cm = 100*x1grid
    x2grid_cm = 100*x2grid
    co = contourf!(ax1, x1grid_cm, x2grid_cm, data, levels=20, colormap=:plasma) #levels = range(0.0, 10.0, length = 20))
    ax1.xticks = 0 : 5 : 30;
    ax1.yticks = 0 : 1 : 5;
    # Colorbar(f[1, 2], co)
    Colorbar(f[1, 2], co, ticks = [300, 301, 302, 303])
    f    

   #save(filename, f, pt_per_unit = 1)   
end

begin
    data = reshape(temp_data[:,61],N₁,N₂)
    filename = path2folder*"feedforward_temp_distribution_60.pdf"
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = L"Length $x_{1}$ in [cm]", ylabel = L"Width $x_{2}$ in [cm]", xlabelsize = 30, ylabelsize = 30)
    # tightlimits!(ax1)
    # hidedecorations!(ax1)
    x1grid_cm = 100*x1grid
    x2grid_cm = 100*x2grid
    co = contourf!(ax1, x1grid_cm, x2grid_cm, data, levels=20, colormap=:plasma) #levels = range(0.0, 10.0, length = 20))
    ax1.xticks = 0 : 5 : 30;
    ax1.yticks = 0 : 1 : 5;
    Colorbar(f[1, 2], co)
    f    

   save(filename, f, pt_per_unit = 1)   
end

begin
    data = reshape(temp_data[:,91],N₁,N₂)
    filename = path2folder*"feedforward_temp_distribution_90.pdf"
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = L"Length $x_{1}$ in [cm]", ylabel = L"Width $x_{2}$ in [cm]", xlabelsize = 30, ylabelsize = 30)
    # tightlimits!(ax1)
    # hidedecorations!(ax1)
    x1grid_cm = 100*x1grid
    x2grid_cm = 100*x2grid
    co = contourf!(ax1, x1grid_cm, x2grid_cm, data, levels=20, colormap=:plasma) #levels = range(0.0, 10.0, length = 20))
    ax1.xticks = 0 : 5 : 30;
    ax1.yticks = 0 : 1 : 5;
    Colorbar(f[1, 2], co)
    f    

   save(filename, f, pt_per_unit = 1)   
end

begin
    data = reshape(temp_data[:,121],N₁,N₂)
    filename = path2folder*"feedforward_temp_distribution_120.pdf"
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = L"Length $x_{1}$ in [cm]", ylabel = L"Width $x_{2}$ in [cm]", xlabelsize = 30, ylabelsize = 30)
    # tightlimits!(ax1)
    # hidedecorations!(ax1)
    x1grid_cm = 100*x1grid
    x2grid_cm = 100*x2grid
    co = contourf!(ax1, x1grid_cm, x2grid_cm, data, levels=20, colormap=:plasma) #levels = range(0.0, 10.0, length = 20))
    ax1.xticks = 0 : 5 : 30;
    ax1.yticks = 0 : 1 : 5;
    Colorbar(f[1, 2], co)
    f    

   save(filename, f, pt_per_unit = 1)   
end


begin
    data = temp_data[(N₂-1)*N₁+1:N₂*N₁,:]
    filename = path2folder*"feedforward_temperature_north_contour_complete.pdf"
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = L"Length $x_{1}$ in [cm]", ylabel = "Time in [s]", xlabelsize = 30, ylabelsize = 30)
    # tightlimits!(ax1)
    # hidedecorations!(ax1)
    x1grid_cm = 100*x1grid
    #x2grid_cm = 100*x2grid
    co = contourf!(ax1, x1grid_cm, sol_final.t, data, levels=40, colormap=:plasma) #levels = range(0.0, 10.0, length = 20))
    ax1.xticks = 0 : 5 : 30;
    #ax1.yticks = 0 : 1 : 5;
    Colorbar(f[1, 2], co, ticks = [300, 350, 400, 450,500])
    f    

    save(filename, f, pt_per_unit = 1)   
end

begin
    data = temp_data[(N₂-1)*N₁+1:N₂*N₁,61:end]
    filename = path2folder*"feedforward_temperature_north_contour_2nd_part.pdf"
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = L"Length $x_{1}$ in [cm]", ylabel = "Time in [s]", xlabelsize = 30, ylabelsize = 30)
    # tightlimits!(ax1)
    # hidedecorations!(ax1)
    x1grid_cm = 100*x1grid
    #x2grid_cm = 100*x2grid
    co = contourf!(ax1, x1grid_cm, sol_final.t[61:end], data, levels=20, colormap=:plasma) #levels = range(0.0, 10.0, length = 20))
    ax1.xticks = 0 : 5 : 30;
    #ax1.yticks = 0 : 1 : 5;
    Colorbar(f[1, 2], co, ticks = [400, 425, 450, 475,500])
    f    

    save(filename, f, pt_per_unit = 1)   
end

begin   
    filename = path2folder*"feedforward_temperature_north.pdf"
    data1 = temp_data[(N₂-1)*N₁+1:N₂*N₁,81]
    data2 = temp_data[(N₂-1)*N₁+1:N₂*N₁,101]
    data3 = temp_data[(N₂-1)*N₁+1:N₂*N₁,121]


    f = Figure(size=(600,400),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel = L"Length $x_{1}$ in [cm]", ylabel="Temperature in [K]",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    
    #ax1.xticks = 0 : 200 : 1200;
    #ax1.yticks = 0 : 2.5 : 10;

    tstart = 450;
    tend = 2550;
    x1grid_cm = 100*x1grid
    lines!(x1grid_cm, data1;   linestyle = :dot,   linewidth = 5, label="t=60 s")
    lines!(x1grid_cm, data2;   linestyle = :dash,   linewidth = 5, label="t=90 s")
    lines!(x1grid_cm, data3;   linestyle = :dash,   linewidth = 5, label="t=120 s")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);
    

    f  
    #save(filename, f, pt_per_unit = 1)   
end



begin
    fig = Figure(size=(800,600),fontsize=26)
    ax = Axis3(fig[1,1], azimuth = 5pi/4, 
                xlabel = "Time t in [s]", ylabel = "Position x in [m]", zlabel = "Temperature in [K]", 
                xlabelsize = 30,  ylabelsize = 30)

    surface!(ax,  xgrid, sol.t[1:10:end], Array(sol_final)[(N₂-1)*N₁+1:N₂*N₁,1:10:end], colormap = :plasma)            
    # surface!(ax,  xgrid, sol.t, Array(sol_final)[(N₂-1)*N₁+1:N₂*N₁,:], colormap = :plasma)            
    fig
    #save("results/figures/"*"temp_north_3d.pdf", fig,pt_per_unit = 1)    
end
