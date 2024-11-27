L = 0.1; # Length of 1D rod
# Aluminium
λ = 50;  # Thermal conductivity
ρ = 8000; # Density
c = 400;  # Specific heat capacity
α = λ / (ρ * c) # Diffusivity

Nx = 5;
Δx = L/Nx

D1 = zeros(Int64,Nx, Nx)
for i=2:Nx-1
    D1[i,i-1 : i+1] = [1,-2,1];
end
D1[1,1:2] = [-1,1]
D1[Nx,Nx-1:Nx] = [1,-1]

a1 = α/(Δx^2)
a1_inv = (Δx^2)/α 
A = α*D1/(Δx^2)
b1 = (Δx*c*ρ)
B = vcat(1, zeros(Int64,Nx-1)) / b1

function input_triangle(t)
    a = 100;
    if t < Tf/2
        return a*t
    else
        return a*(Tf-t)
    end
end


function input_bang(t)
    a = 100;
    if t < Tf/2
        return 4e4
    else
        return 0
    end
end

Tf = 1000;
tgrid = 0 :1: Tf
using Plots
plot(tgrid, input_triangle.(tgrid))

# 1D heat equation
function heat_eq!(dx,x,p,t)       
    # time = t/Tf;
    #u = input_signal(time, p)
    
    # u_in = input_triangle(t)
    u_in = 0# input_bang(t)

    dx .= A*x + B*u_in
end


const ts = Tf/1000   # Time step width

# Simulation without optimization
using OrdinaryDiffEq

x0 = 300*(1 .+ 3*rand(Nx)) # 300 * ones(Nx) # Intial values
tspan = (0.0, Tf)   # Time span
alg = KenCarp4()    # Numerical integrator

prob = ODEProblem(heat_eq!,x0,tspan)
sol = solve(prob,alg, saveat = ts)

plot(sol)

plot(sum(A*Array(sol),dims=1)')

plot((A*Array(sol))')

Mgrad = zeros(Int64,Nx-1, Nx)
for i=1:Nx-1
    Mgrad[i,i: i+1] = [-1,1] ./ Δx;
end

plot((Mgrad*Array(sol))')

Mgrad2 = zeros(Int64,Nx-2, Nx-1)
for i=1:Nx-2
    Mgrad2[i,i: i+1] = [1,-1] ./ Δx;
end

plot((Mgrad'*Mgrad*Array(sol))')