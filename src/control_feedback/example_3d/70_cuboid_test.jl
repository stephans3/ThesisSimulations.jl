
θvec = collect(300:50:500)
M_temp = mapreduce(z-> [1  z  z^2 z^3 z^4], vcat, θvec)

λ1data = [40,44,50,52,52.5]
λ1p = inv(M_temp)*λ1data

λ3data = [40,55,60,65,68]
λ3p = inv(M_temp)*λ3data

ρ = 8000; # Density
c = 400;  # Specific heat capacity

L = 0.2; # Length 
W = 0.2; # Width
H = 0.05;# Height
N₁ = 10;
N₂ = 10;
N₃ = 5;
Nc = N₁*N₂*N₃ 
Δx₁ = L/N₁
Δx₂ = W/N₂
Δx₃ = W/N₃


using Hestia 

property = DynamicAnisotropic(λ1p, λ1p, λ3p, [ρ],[c])
cuboid  = HeatCuboid(L,W, H, N₁,N₂,N₃)
boundary = Boundary(cuboid)

### Boundaries ###
Θamb = 300.0;
h = 10;
ε = 0.1;
emission = Emission(h, Θamb,ε) #  Stefan-Boltzmann BC: heat transfer (linear) and heat radiation (quartic/nonlinear)
boundary = Boundary(cuboid)

setEmission!(boundary, emission, :east )
setEmission!(boundary, emission, :south )
setEmission!(boundary, emission,  :topside )

### Actuation ###
actuation = IOSetup(cuboid)
num_actuators = (2,2)        # Number of actuators per boundary
config  = RadialCharacteristics(1.0, 2, 30)
# config  = RadialCharacteristics(1.0, 2, 0)
setIOSetup!(actuation, cuboid, num_actuators, config, :underside)



function heat_conduction!(dθ, θ, param, t)
    u1 = 1e5;
    u_in = u1*ones(4);
    diffusion!(dθ, θ, cuboid, property, boundary, actuation, u_in)
end



Tf = 1200;
tspan = (0.0, Tf)
θ₀ = 300;
θinit = θ₀*ones(Nc)
dΘ = similar(θinit)
#heat_conduction!(dΘ,θinit,pinit,0)

using OrdinaryDiffEq

t_samp = 30.0
alg = KenCarp5()
prob = ODEProblem(heat_conduction!,θinit,tspan)
sol = solve(prob,alg, saveat=t_samp)


### Sensor ###
num_sensor = (2,2)        # Number of sensors
Ny = num_sensor[1]*num_sensor[2]
sensing = IOSetup(cuboid)
config_sensor  = RadialCharacteristics(1.0, 1, 0)
setIOSetup!(sensing, cuboid, num_sensor, config_sensor, :topside)


C = zeros(Ny,Nc)
b_sym = :topside
ids = unique(sensing.identifier[b_sym])
for i in ids
    idx = findall(x-> x==i, sensing.identifier[b_sym])
    boundary_idx = sensing.indices[b_sym]
    boundary_char = sensing.character[b_sym]
    C[i,boundary_idx[idx]] = boundary_char[idx] / sum(boundary_char[idx])
end

yout = C*Array(sol)

data_end = reshape(Array(sol[end]),N₁,N₂,N₃)
data_end[:,:,end]

using Plots
contourf(data_end[:,:,end])

Cc = sensor_char' ./  sum(sensor_char, dims=1)'

C

yout[:,end]
sum(data_end[1:5,1:5,end])/25
sum(data_end[6:10,1:5,end])/25
sum(data_end[1:5,6:10,end])/25
sum(data_end[6:10,6:10,end])/25