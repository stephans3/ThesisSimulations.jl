
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


# Reference
ref_init = 300;
Δr = 200; # Difference operating points
ps = 10; # Steepness
ψ(t) = (1+tanh(10*(t/Tf - 1/2)))/2
ref(t) = ref_init + Δr*ψ(t)


function heat_conduction!(dθ, θ, param, t)

    u1 = input_obc(t,param[1:3])
    u2 = input_obc(t,param[4:6])
    
    u_in = [u1, u2, u1]
    diffusion!(dθ, θ, plate, property, boundary, actuation, u_in)
end



Tf = 1200;
tspan = (0.0, Tf)
θ₀ = 300;
Ntotal = N₁*N₂
θinit = θ₀*ones(Ntotal)

dΘ = similar(θinit)


p_energy = [12.394207145215864
        2.070393374741201
        9.149105824173542]

pinit = repeat(p_energy,2)

heat_conduction!(dΘ,θinit,pinit,0)

# Δt = 1e-2               # Sampling time

using OrdinaryDiffEq

t_samp = 30.0
alg = KenCarp5()
prob = ODEProblem(heat_conduction!,θinit,tspan,pinit)
sol = solve(prob,alg,p=pinit, saveat=t_samp)
ntime = length(sol.t)

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

# loss = sum(abs2, err) / Tf
loss = sum(abs2, err)*t_samp / Tf


function loss_optim(u,p)
    pars = u
    sol_loss = solve(prob, alg,p=pars, saveat = t_samp)
    if sol_loss.retcode == ReturnCode.Success
        y = C*Array(sol_loss)
        err = ref_data - y
        loss = sum(abs2, err) *t_samp / Tf
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


p_opt = repeat(p_energy,2)
loss_optim(p_opt, [0])

using Optimization, OptimizationOptimJL, ForwardDiff
optf = OptimizationFunction(loss_optim, Optimization.AutoForwardDiff())
opt_prob = OptimizationProblem(optf, p_opt, [0])
p_opt = Optimization.solve(opt_prob, ConjugateGradient(), callback=callback, maxiters=30)

#=
p_opt = [
12.317591757756611
  2.0563320468272157
  8.1946771566001
 12.191130635950069
  2.0895363718616418
  8.717129140569332]
=#

loss_optim(p_opt, [0])

st_pars = hcat(store_param...)

#=
st_pars=[
    12.3942  2.07039  9.14911  12.3942  2.07039  9.14911
    12.3598  2.05982  9.15227  12.3765  2.06541  9.15074
    12.3627  2.0618   9.15164  12.3774  2.06647  9.15048
    12.3633  2.0627   9.15124  12.3771  2.06702  9.15033
    12.3638  2.0641   9.15041  12.376   2.06797  9.15005
    12.3635  2.06629  9.148    12.3716  2.0699   9.14924
    12.3729  2.06112  9.12985  12.3494  2.07382  9.14276
    12.3745  2.0613   9.12731  12.348   2.07437  9.14173
    12.3757  2.06184  9.11048  12.3392  2.07634  9.1346
    12.3752  2.06042  9.0779   12.325   2.07811  9.12047
    12.3802  2.05936  9.01951  12.3052  2.08138  9.09493
    12.3804  2.05755  8.9756   12.2946  2.0823   9.07528
    12.3732  2.05266  8.84892  12.2691  2.08344  9.01797
    12.3636  2.04983  8.60996  12.2288  2.08697  8.9094
    12.3436  2.05615  8.39745  12.2029  2.09073  8.81168
    12.334   2.05642  8.32253  12.1979  2.09047  8.77679
    12.3184  2.05635  8.20125  12.1914  2.0896   8.7202
    12.3176  2.05633  8.19468  12.1911  2.08954  8.71713
    12.3176  2.05633  8.19468  12.1911  2.08954  8.71713
    12.3176  2.05633  8.19468  12.1911  2.08954  8.71713
    12.3176  2.05633  8.19468  12.1911  2.08954  8.71713
    12.3176  2.05633  8.19468  12.1911  2.08954  8.71713]'
=#

#=
loss_store =[85.62637828590933
            39.95139125822754
            39.27330111588842
            39.18914386498823
            39.09525464837438
            38.93776164478812
            38.04789725292938
            38.00549263364592
            37.663897299603576
            37.31792207130319
            36.589327301594324
            36.11124996716428
            34.83005537538507
            32.93886479752774
            28.61599393033453
            27.636085683029638
            26.143589231132164
            26.04844222982186
            26.048442211397685
            26.048442204595954
            26.04844219535964
            26.04844219535964]
=#


hcat(store_param...)[1,:]
hcat(store_param...)[4,:]

hcat(hcat(store_param...)[2,:],hcat(store_param...)[5,:])
hcat(hcat(store_param...)[3,:],hcat(store_param...)[6,:])



using CairoMakie
path2folder = "results/figures/controlled/"
begin
    filename = path2folder*"feedforward_obc_parameter_1.pdf"
    pars_data = hcat(st_pars[1,:],st_pars[4,:])
    #pars_data = hcat(hcat(store_param...)[1,:],hcat(store_param...)[4,:])

    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Iteration Number", ylabel = L"Parameter $p_{1}$", xlabelsize = 30, ylabelsize = 30)
    
    num_iterations = 0:length(pars_data[:,1])-1;
    
    scatterlines!(num_iterations, pars_data[:,1], linestyle = :dash,  linewidth = 3, markersize=20, label="Actuator 1")
    scatterlines!(num_iterations, pars_data[:,2], linestyle = :dash,  linewidth = 3, marker = :xcross, markersize=20, label="Actuator 2")

    # lines!(num_iterations, pars_data[:,1], linestyle = :dash,  linewidth = 3)
    # lines!(num_iterations, pars_data[:,2], linestyle = :dash,  linewidth = 3)
    axislegend(; position = :lb, backgroundcolor = (:grey90, 0.1), labelsize=30);
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
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);
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
    axislegend(; position = :lb, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f

    save(filename, f, pt_per_unit = 1)   
end

begin
    filename = path2folder*"feedforward_obc_loss.pdf"
    # pars_data = hcat(st_pars[3,:],st_pars[6,:])
    
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Iteration Number", ylabel = "Objective", xlabelsize = 30, ylabelsize = 30)
    
    num_iterations = 0:length(pars_data[:,1])-1;
    
    scatterlines!(num_iterations, store_loss, linestyle = :dash,  linewidth = 3, markersize=20)
    # scatterlines!(num_iterations, pars_data[:,2], linestyle = :dash,  linewidth = 3, marker = :xcross, markersize=20, label="Actuator 2")

    # lines!(num_iterations, pars_data[:,1], linestyle = :dash,  linewidth = 3)
    # lines!(num_iterations, pars_data[:,2], linestyle = :dash,  linewidth = 3)
    # axislegend(; position = :rt, backgroundcolor = (:grey90, 0.1), labelsize=30);

    ax2 = Axis(f, bbox=BBox(250, 570, 220, 375), ylabelsize = 24) #(:grey90,0.1))
    ax2.xticks = 10 : 5 : 20;
    ax2.yticks = [26, 29, 32,35];
    scatterlines!(ax2, num_iterations[11:end], store_loss[11:end], linestyle = :dash,  linewidth = 3, markersize=20)
    #lines!(ax2, num_iterations[5:end], store_loss[5:end];   linestyle = :dash,  linewidth = 5, color=Makie.wong_colors()[1])
    translate!(ax2.scene, 0, 0, 10);
    #translate!(ax2.elements[:background], 0, 0, 9)
    elements = keys(ax2.elements)
    filtered = filter(ele -> ele != :xaxis && ele != :yaxis, elements)
    foreach(ele -> translate!(ax2.elements[ele], 0, 0, 9), filtered)
    f

    save(filename, f, pt_per_unit = 1)   
end






sol_opt = solve(prob,alg,p=p_opt, saveat= 10) #t_samp)
y_opt = C*Array(sol_opt)

input1(t) = input_obc(t,p_opt[1:3])
input2(t) = input_obc(t,p_opt[4:6])

input1_data = input1.(sol_opt.t)
input2_data = input2.(sol_opt.t)

ref_data_2 = ref.(sol_opt.t)

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
    lines!(sol_opt.t, scale*input1_data;   linestyle = :dot,   linewidth = 5, label="Input 1")
    lines!(sol_opt.t, scale*input2_data;   linestyle = :dash,   linewidth = 5, label="Input 2")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);
    

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
    lines!(sol_opt.t, y_opt[1,:];   linestyle = :dot,   linewidth = 5, label="Output 1")
    lines!(sol_opt.t, y_opt[2,:];   linestyle = :dash,   linewidth = 5, label="Output 2")
    scatter!(sol_opt.t[1:15:end], ref_data_2[1:15:end];  markersize = 15, marker = :diamond, color=:black, label = "Reference")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);
     
    
    ax2 = Axis(f, bbox=BBox(410, 560, 140, 260), ylabelsize = 24)
    ax2.xticks = 900 : 150 : Tf;
    ax2.yticks = [495, 500];
    lines!(sol_opt.t[91:end], y_opt[1,91:end];   linestyle = :dot,   linewidth = 5)
    lines!(sol_opt.t[91:end], y_opt[2,91:end];   linestyle = :dash,   linewidth = 5)
    scatter!(sol_opt.t[91:15:end], ref_data_2[91:15:end];  markersize = 15, marker = :diamond, color=:black, label = "Reference")
    translate!(ax2.scene, 0, 0, 10);
    elements = keys(ax2.elements)
    filtered = filter(ele -> ele != :xaxis && ele != :yaxis, elements)
    foreach(ele -> translate!(ax2.elements[ele], 0, 0, 9), filtered)
    
    f  
    save(filename, f, pt_per_unit = 1)   
end

x1grid = Δx₁/2 : Δx₁ : L
x2grid = Δx₂/2 : Δx₂ : W

Array(sol_opt)[(N₂-1)*N₁+1:N₂*N₁,1:20:end]
temp_data = Array(sol_opt)

# filename = path2folder*"feedforward_temp_distribution_30.pdf"
begin
    data = reshape(temp_data[:,31],N₁,N₂)
    filename = path2folder*"feedforward_temp_distribution_30.pdf"
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = L"Length $x_{1}$ in [cm]", ylabel = L"Width $x_{2}$ in [cm]", xlabelsize = 30, ylabelsize = 30)
    tightlimits!(ax1)
    # hidedecorations!(ax1)
    x1grid_cm = 100*x1grid
    x2grid_cm = 100*x2grid
    co = contourf!(ax1, x1grid_cm, x2grid_cm, data, levels = 300:0.1:302, colormap=:plasma) #levels = range(0.0, 10.0, length = 20))
    ax1.xticks = 0 : 5 : 30;
    ax1.yticks = 0 : 1 : 5;
    # Colorbar(f[1, 2], co)
    Colorbar(f[1, 2], co ,  ticks = [300, 301, 302])
    f    

    save(filename, f, pt_per_unit = 1)   
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
    Colorbar(f[1, 2], co,  ticks = [400, 425, 450, 475])
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
    Colorbar(f[1, 2], co,  ticks = [497, 500, 503])
    f    

    save(filename, f, pt_per_unit = 1)   
end

begin
    data = reshape(temp_data[:,121],N₁,N₂)
    filename = path2folder*"feedforward_temp_distribution_120.pdf"
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = L"Length $x_{1}$ in [cm]", ylabel = L"Width $x_{2}$ in [cm]", xlabelsize = 30, ylabelsize = 30)
    tightlimits!(ax1)
    # hidedecorations!(ax1)
    x1grid_cm = 100*x1grid
    x2grid_cm = 100*x2grid
    co = contourf!(ax1, x1grid_cm, x2grid_cm, data, levels = 493:0.15:496, colormap=:plasma) #levels = range(0.0, 10.0, length = 20))
    ax1.xticks = 0 : 5 : 30;
    ax1.yticks = 0 : 1 : 5;
    Colorbar(f[1, 2], co,  ticks = [493, 494, 495,496])
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
    co = contourf!(ax1, x1grid_cm, sol_opt.t, data, levels=40, colormap=:plasma) #levels = range(0.0, 10.0, length = 20))
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
    co = contourf!(ax1, x1grid_cm, sol_opt.t[61:end], data, levels=20, colormap=:plasma) #levels = range(0.0, 10.0, length = 20))
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




















plot(hcat(store_param...)[1,:])
plot!(hcat(store_param...)[4,:])




sol_obc = solve(prob,alg,p=p_opt, saveat=t_samp)
plot(sol_obc)




using Plots
plot(sol_obc)













function loss_optim_p2(u,p)
    pars = [p[1],u[1],p[2],p[1],u[2],p[2]]
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


p_opt = repeat([p_energy[2]],2)
p_else = [p_energy[1],p_energy[3]]
loss_optim_p2(p_opt, p_else)

using Optimization, OptimizationOptimJL, ForwardDiff
optf = OptimizationFunction(loss_optim_p2, Optimization.AutoForwardDiff())
opt_prob = OptimizationProblem(optf, p_opt, p_else)
p2_pos = Optimization.solve(opt_prob, ConjugateGradient(), callback=callback, maxiters=15)

p_all = [p_energy[1],p2_pos[1],p_energy[3],p_energy[1],p2_pos[2],p_energy[3]]
sol_obc = solve(prob,alg,p=p_all, saveat=t_samp)

















hcat(store_param...)[1,:]
hcat(store_param...)[4,:]

plot(hcat(store_param...)[1,:])
plot!(hcat(store_param...)[4,:])


sol_obc = solve(prob,alg,p=p_opt, saveat=t_samp)
plot(sol_obc)




using Plots
plot(sol_obc)

scatter(hcat(store_param...)')

y_obc = C*Array(sol_obc)
plot(sol_obc.t, y_obc[1:2,:]')
scatter!(sol_obc.t, ref_data[1,:])


Δp2 = zeros(6);
Δp2[2] = 0.1;
Δp2[5] = -0.1;
pblubb = repeat(p_energy,2) + Δp2



sol_obc = solve(prob,alg,p=p_final, saveat=t_samp)
plot(sol_obc)

y_obc = C*Array(sol_obc)
plot(sol_obc.t, y_obc[1:2,:]')
scatter!(sol_obc.t, ref_data[1,:])

plot(sol_obc.t, y[1:2,:]')
scatter!(sol_obc.t, ref_data[1,:])


plot(hcat(store_param...)[1,:])
plot!(hcat(store_param...)[4,:])

plot(hcat(store_param...)[2,:])
plot!(hcat(store_param...)[5,:])

plot(hcat(store_param...)[3,:])
plot!(hcat(store_param...)[6,:])



Δp2 = zeros(9);
# Δp2[5] = -0.05;
Δp22[2:6:end] .= -0.04;
Δp22
repeat(p_energy,3)+Δp2
p_opt = repeat(p_energy,3)+Δp2
loss_optim_std(p_opt, [0])

# 8.562075292894844



########################








u₀ = 1e-1;
p₂ = p_fbc[2]

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

