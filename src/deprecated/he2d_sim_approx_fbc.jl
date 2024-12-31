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
N₁ = 3;
N₂ = 5;
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
config  = RadialCharacteristics(1.0, 1, 0)
setIOSetup!(actuation, plate, num_actuators, config, :south)

function input_obc(t,p)
    return exp(p[1] - p[3]^2 * (t/Tf - 1/p[2])^2)
end


function heat_conduction!(dθ, θ, param, t)

    u1 = input_obc(t,param[1:3])
    u2 = input_obc(t,param[4:6])
    u3 = input_obc(t,param[7:9])
    
    u_in = [u1, u2, u3]
    diffusion!(dθ, θ, plate, property, boundary, actuation, u_in)
end


Tf = 1200;
tspan = (0.0, Tf)
θ₀ = 300;
Ntotal = N₁*N₂
θinit = θ₀*ones(Ntotal)

dΘ = similar(θinit)

p_found = [11.81522609265776
            2.070393374741201
            9.212088025161702]
pinit = repeat(p_found,3)

heat_conduction!(dΘ,θinit,pinit,0)

# Δt = 1e-2               # Sampling time

using OrdinaryDiffEq

t_samp = 10.0
alg = KenCarp5()
prob = ODEProblem(heat_conduction!,θinit,tspan,pinit)
sol = solve(prob,alg,p=pinit, saveat=t_samp)



input_u(t) = input_obc(t,p_found)
input_data = input_u.(sol.t)
input_int = sum(input_data[2:end])*t_samp
E_in = 0.3*input_int
Δr = 100;
ΔU = ρ *c *L*W*Δr


using Plots
plot(sol)
heatmap(reshape(Array(sol)[:,60],N₁,N₂))



plot(sol.t, input_u.(sol.t))


actuation2 = IOSetup(plate)
config  = RadialCharacteristics(1.0, 3, 20.0)
setIOSetup!(actuation2, plate, num_actuators, config, :south)


### Sensor ###
num_sensor = 3        # Number of sensors
sensing = IOSetup(plate)
config_sensor  = RadialCharacteristics(1.0, 2, 20.0)
setIOSetup!(sensing, plate, num_sensor, config_sensor, :north)
