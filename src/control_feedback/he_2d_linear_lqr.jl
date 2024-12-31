#=
    LQR Design for 2D Heat Conduction
=#
λ₁ = 40;
λ₂ = 60;
ρ = 8000; # Density
c = 400;  # Specific heat capacity

L = 0.3; # Length
W = 0.05
N₁ = 30;
N₂ = 10;
Nc = N₁*N₂ 
Δx₁ = L/N₁
Δx₂ = W/N₂


using Hestia 
property = StaticAnisotropic([λ₁,λ₂], ρ,c)
plate  = HeatPlate(L,W, N₁,N₂)
boundary = Boundary(plate)

### Boundaries ###
θamb = 300.0;
h = 10;
ε = 0.1;
emission = Emission(h, θamb,ε) #  Stefan-Boltzmann BC: heat transfer (linear) and heat radiation (quartic/nonlinear)
boundary = Boundary(plate)

setEmission!(boundary, emission, :west )
setEmission!(boundary, emission, :east )
setEmission!(boundary, emission,  :north )


# Emitted power at initial temperature
Pem_W_approx = (L/3)*emit(500,emission) + W*emit(500,emission)
Pem_E_approx = (L/3)*emit(500,emission) + W*emit(500,emission)
Pem_N_approx = (L/3)*emit(500,emission)
Pem_WEN_approx = Pem_W_approx+ Pem_E_approx+Pem_N_approx

### Actuation ###
actuation = IOSetup(plate)
num_actuators = 3        # Number of actuators per boundary
config  = RadialCharacteristics(1.0, 2, 30)
setIOSetup!(actuation, plate, num_actuators, config, :south)

actuator_char = getCharacteristics(actuation, :south)[1]

# LQR Design
α₁ = λ₁ / (ρ * c) # Diffusivity
α₂ = λ₂ / (ρ * c) # Diffusivity

# Diffusion matrices
D1 = zeros(Int64,Nc,Nc)
D2 = zeros(Int64,Nc,Nc)
for j=1:N₂
    di = (j-1)*N₁
    for i=2:N₁-1
        D1[i+di,i-1+di : i+1+di] = [1,-2,1];
    end
    D1[1+di,1+di:2+di] = [-1,1]
    D1[N₁+di,N₁-1+di:N₁+di] = [1,-1]
end


for i=1:N₁
    for j=2:N₂-1
        di = (j-1)*N₁
        D2[i+di,i+di-N₁:N₁:(i+di)+N₁] =  [1,-2,1]
        #D2[i+(j-1)*N₁,i-1+(j-1)*N₁ : i+1+(j-1)*N₁] = [1,-2,1];
    end
    D2[i,i:N₁:i+N₁] = [-1,1]
    D2[i+(N₂-1)*N₁,i+(N₂-2)*N₁:N₁:i+(N₂-1)*N₁] = [1,-1]
end

a1 = α₁/(Δx₁^2);
a2 = α₂/(Δx₂^2);

using LinearAlgebra
A = a1*D1 + a2*D2
b1 = (Δx₂*c*ρ)
B = vcat(actuator_char, zeros(N₁*(N₂-1),num_actuators)) / b1;

# Controller Design with
# Linear Quadratic Regulator
Rw = diagm(ones(num_actuators));
Qw =  diagm(1e8*ones(N₁*N₂)) # diagm(mapreduce(i->i*1e7*ones(N₁),vcat, 1:1:N₂))
Sw = zeros(N₁*N₂, num_actuators);

# Solving Riccati Equation
using MatrixEquations
P,evals_cl,K = arec(A, B, Rw, Qw, Sw)

# Acl = A-B*K


# Heat Conduction Simulation
function heat_conduction!(dθ, θ, param, t)
    u_in =  -K*(θ-θinit) # zeros(3); # [u1, u2, u3]
    diffusion!(dθ, θ, plate, property, boundary, actuation, u_in)
end

Tf = 600 
tspan = (0.0, Tf)
θ₀ = 500;
Ntotal = N₁*N₂
θinit = θ₀*ones(Ntotal)

dΘ = similar(θinit)
heat_conduction!(dΘ,θinit,0,0)

using OrdinaryDiffEq
t_samp = 10.0
alg = KenCarp5()
prob = ODEProblem(heat_conduction!,θinit,tspan)
sol = solve(prob,alg, saveat=t_samp)
θsol = Array(sol)


# Sensor setup
num_sensor = 3        # Number of sensors
sensing = IOSetup(plate)
config_sensor  = RadialCharacteristics(1.0, 1, 0)
setIOSetup!(sensing, plate, num_sensor, config_sensor, :north)

sensor_char = getCharacteristics(sensing, :north)[1]
C = hcat(zeros(3,N₁*(N₂-1)), sensor_char' ./  sum(sensor_char, dims=1)')


using CairoMakie
path2folder = "results/figures/controlled/feedback/"
begin
    udata = -K*(θsol.-θinit)
    filename = path2folder*"lqr_input.pdf"

    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Time in [s]", ylabel = L"Input $\times 10^{3}$", xlabelsize = 30, ylabelsize = 30)
    
    scale = 1e-3;

    lines!(sol.t, scale*udata[1,:], linestyle = :dot,  linewidth = 5, label="Actuator 1")
    lines!(sol.t, scale*udata[2,:], linestyle = :dash,  linewidth = 5, label="Actuator 2")

    axislegend(; position = :rb, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f

    save(filename, f, pt_per_unit = 1)   
end


begin
    yout = C*θsol
    filename = path2folder*"lqr_output.pdf"

    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Time in [s]", ylabel = L"Output in [K] $~$", xlabelsize = 30, ylabelsize = 30)
    
    scale = 1;

    lines!(sol.t, scale*yout[1,:], linestyle = :dot,  linewidth = 5, label="Sensor 1")
    lines!(sol.t, scale*yout[2,:], linestyle = :dash,  linewidth = 5, label="Sensor 2")

    ax1.yticks = [498.5, 499, 499.5, 500];
    axislegend(; position = :rt, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f

    save(filename, f, pt_per_unit = 1)   
end



# Emitted Power
n_ts = length(sol.t)
θsol_2d = reshape(θsol, N₁,N₂,n_ts)
ϕem_W = map(θ-> emit(θ,emission), θsol_2d[1,1:N₂,:])
ϕem_E = map(θ-> emit(θ,emission), θsol_2d[N₁,1:N₂,:])
ϕem_N = map(θ-> emit(θ,emission), θsol_2d[1:N₁,N₂,:])

Pem_W = Δx₂*sum(ϕem_W,dims=1)[:]
Pem_E = Δx₂*sum(ϕem_E,dims=1)[:]
Pem_N = Δx₁*sum(ϕem_N,dims=1)[:]

Pem = Pem_W + Pem_E + Pem_N

# Approx. emitted power
Pem_approx = L*emit(500,emission) + 2*W*emit(500,emission)

# Supplied power
Pin = Δx₁*sum(vcat(actuator_char, zeros(N₁*(N₂-1),num_actuators))*udata,dims=1)[:]

begin
    filename = path2folder*"lqr_power.pdf"

    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Time in [s]", ylabel = L"Power in [W] $~$", xlabelsize = 30, ylabelsize = 30, limits = (nothing, (-50, 1050)),)
    
    scale = 1;

    lines!(sol.t, scale*abs.(Pem), linestyle = :dot,  linewidth = 5, label="Abs. Emitted")
    lines!(sol.t, scale*Pin, linestyle = :dash,  linewidth = 5, label="Supplied")

    ax1.yticks = [0, 200, 400, 600, 800, 1000];
    # lines!(num_iterations, pars_data[:,1], linestyle = :dash,  linewidth = 5)
    # lines!(num_iterations, pars_data[:,2], linestyle = :dash,  linewidth = 5)
    axislegend(; position = :rb, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f

    save(filename, f, pt_per_unit = 1)   
end


x1grid = Δx₁/2 : Δx₁ : L
x2grid = Δx₂/2 : Δx₂ : W

begin
    data = θsol_2d[:,:,end]
    filename = path2folder*"lqr_temp_contour.pdf"
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = L"Length $x_{1}$ in [cm]", ylabel = L"Width $x_{2}$ in [cm]", xlabelsize = 30, ylabelsize = 30)
    tightlimits!(ax1)
    # hidedecorations!(ax1)
    x1grid_cm = 100*x1grid
    x2grid_cm = 100*x2grid
    co = contourf!(ax1, x1grid_cm, x2grid_cm, data, colormap=:plasma, levels = 496:0.5:502) #levels = range(0.0, 10.0, length = 20))
    ax1.xticks = 0 : 5 : 30;
    ax1.yticks = 0 : 1 : 5;
    Colorbar(f[1, 2], co)
    #Colorbar(f[1, 2], co ,  ticks = [300, 301, 302])
    f    

    save(filename, f, pt_per_unit = 1)   
end






using Plots
plot(sol,legend=false)

plot(θsol[N₁*(N₂-1)+1:end,:]',legend=false)

n_ts = length(sol.t)
θsol_2d = reshape(θsol, N₁,N₂,n_ts)
contourf(θsol_2d[:,:,end])




#


using Plots
plot(sol,legend=false)
plot(θsol[N₁*(N₂-1)+1:end,:]',legend=false)



sum(θsol[:,end])/300




contourf(θsol_2d[:,N₂,:])
heatmap(θsol_2d[:,:,end])

surface(θsol_2d[:,N₂,:])



# plot(Pem)
# plot(Pem_W)



u1 = -Pem_W_approx / (Δx₁*sum(vcat(actuator_char, zeros(N₁*(N₂-1),num_actuators))[:,1]))
u2 = -Pem_N_approx / (Δx₁*sum(vcat(actuator_char, zeros(N₁*(N₂-1),num_actuators))[:,2]))

[u1,u2,u1]
# [5845.586245698477, 3897.0574971323185, 5845.586245698477]
Δx₁*sum(vcat(actuator_char, zeros(N₁*(N₂-1),num_actuators))[:,1])*3900

Pem_approx = L*emit(500,emission) + 2*W*emit(500,emission)

Δx₁*sum(vcat(actuator_char, zeros(N₁*(N₂-1),num_actuators))*[6220,3150,6220],dims=1)[:]
Δx₁*sum(vcat(actuator_char, zeros(N₁*(N₂-1),num_actuators))*[5197,5197,5197],dims=1)[:]

Pin = Δx₁*sum(vcat(actuator_char, zeros(N₁*(N₂-1),num_actuators))*udata,dims=1)[:]
plot(Pin)
using Plots

plot(udata')

plot(θsol[N₁*(N₂-1)+1:end,:]')


plot(sol)
contourf(reshape(sol[end],N₁,N₂))