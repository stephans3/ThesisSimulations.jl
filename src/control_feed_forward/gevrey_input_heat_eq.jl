

using BellBruno
N_der = 20;              # Number of derivatives

# Either create new Bell polynomials  ones
bp = bell_poly(N_der);   # Create bell polynomials

# Or load already existing Bell polynomial numbers
# bp_load = BellBruno.read_bell_poly()
# bp = bp_load[1:N_der]

# Compute Bell coefficients
bc = bell_coeff(bp);     # Compute bell coefficients

#=
    f(z) = -1 z^(-2)
    d^n/dz^n f(z) = -1 * z^(-2-n) π(-2-j) for j=0:1:n
=#
function outer_fun!(y, z, p) 
    c = -1;

    y[1] = simple_monomial(z, c, p)

    fi = firstindex(y)
    li = lastindex(y)

    for idx in fi:li-1
        y[idx+1] = simple_monomial_der(z, c, p, idx)
    end
end

#=
    g(t)  = t/T - (t/T)^2
    g'(t) = 1/T - 2*t/T^2
    g''(t) = -2/T^2
    g^(n)(t) = 0 for n>2
=#
function inner_fun!(x, t :: Float64; T = 1.0 :: Real)
    c₁ = 0;
    c₂ = 1/T;
    c₃ = -1/T^2;

    x[1] = c₁ + c₂*t + c₃*t^2;  # g(t)  = t/T - (t/T)^2
    x[2] = c₂ + 2*c₃*t;         # g'(t) = 1/T - (2/T^2)*t
    x[3] = 2*c₃;                # g''(t)= -2/T^2
    x[4:end] .= 0;              # g^(n)(t) = 0 for n>2
end



function build_derivative(n_der, bc, bp, data_inner, data_outer, tgrid)
    nt = size(data_inner)[1]
    res = zeros(nt)

    data_out_is_vector = false;

    if length(data_outer[1,:]) == 1
        data_out_is_vector = true
    end

    for k=1 : n_der
        fi = firstindex(bp[n_der+1][k][1,:])           
        li = lastindex(bp[n_der+1][k][1,:])
        sol_prod = zeros(BigFloat,nt)   # Solution of the product π
        for μ = fi : li
                
            sol_prod_temp = zeros(BigFloat,nt)
                
            a = bc[n_der+1][k][μ]   # Coefficients
                
            for (idx, _) in enumerate(tgrid)
                @views x = data_inner[idx,:]
                sol_prod_temp[idx] = a * mapreduce(^, *, x, bp[n_der+1][k][:,μ])
            end
            sol_prod = sol_prod + sol_prod_temp
        end

        if data_out_is_vector == true
            res = res + data_outer.*sol_prod
        else
            res = res + data_outer[:,k+1].*sol_prod
        end
    end

    return res
end


function compute_derivatives(n_der, bc, bp, T, dt; w=2)
    tgrid = dt : dt : T-dt; # Time grid
    nt = length(tgrid)      # Number of time steps
    
    # Outer derivatives
    g̃ = zeros(nt, n_der+1); # g̃_n(t) := d^n/dt^n g(t)
    f̃ = zeros(nt, n_der+1); # f̃_n(t) := d^n/dy^n f(z)
    
    for (idx, elem) in enumerate(tgrid)
        @views x = g̃[idx,:]
        @views y = f̃[idx,:]
        inner_fun!(x, elem, T=T)
        outer_fun!(y, x[1], -w) 
    end
    
    q = zeros(nt, n_der);
    h = zeros(BigFloat,nt, n_der+1);
    h[:,1] = exp.(big.(f̃[:,1]))
     
    for n = 1 : n_der
        q[:,n] = build_derivative(n, bc, bp, g̃[:,2:end], f̃, tgrid)
        h[:,n+1] = build_derivative(n, bc, bp, q, h[:,1], tgrid)
        println("Iteration n= ", n)
    end

    return h
end

p_cntr = 2.0
Tf = 400 # 1200 # 3000;    # Final simulation time
N_dt = 1000;
dt = Tf/N_dt; # Sampling time
h_data = compute_derivatives(N_der, bc, bp, Tf, dt, w=p_cntr);

#=
- Compute input signal u(t) for aluminum and steel 38Si7
=#

L = 0.1; # Length of 1D rod

# Aluminium
λ = 50;  # Thermal conductivity
ρ = 8000; # Density
c = 400;  # Specific heat capacity
α = λ / (ρ * c) # Diffusivity
γ = L^2 / α

η(L,α,i) = BigFloat(L)^(2i+1) / (BigFloat(α)^(i+1) * factorial(big(2i+1)))

idx_grid = 0:N_der;
eta_data = zeros(BigFloat, length(idx_grid))

for (n, iter) in enumerate(idx_grid)
    eta_data[n] = η(L,α,iter)
end

using FastGaussQuadrature

bump(t) = exp(-1 / (t/Tf - (t/Tf)^2)^p_cntr)
t_gq, weights_gq = FastGaussQuadrature.gausslegendre(1000)
tshift = Tf/2;
ω_int = tshift *FastGaussQuadrature.dot( weights_gq ,bump.(tshift*t_gq .+ tshift))

diff_ref = 100; # (y_f - y_0) = 100 Kelvin

u_raw = λ*(diff_ref/ω_int) * (h_data * eta_data)


###################################



function input_signal(t,u_data)
    if t <= 0
        return u_data[1]
    elseif t >= Tf
        return u_data[end]
    end
    dt = Tf/(length(u_data)-1)
    τ = t/dt + 1
    t0 = floor(Int, τ)
    t1 = t0 + 1;

    u0 = u_data[t0]
    u1 = u_data[t1]

    a = u1-u0;
    b = u0 - a*t0

    return a*τ + b;
end


# Diffusion: x-direction
function diffusion_x!(dx,x,Nx, Ny, Δx) # in-place
    
    for iy in 1 : Ny
        for ix in 2 : Nx-1
            i = (iy-1)*Nx + ix
            dx[i] =  (x[i-1] - 2*x[i] + x[i+1])/Δx^2
        end
        i1 = (iy-1)*Nx + 1      # West
        i2 = (iy-1)*Nx + Nx     # East
        dx[i1] = (-2*x[i1] + 2*x[i1+1])/Δx^2
        dx[i2] = (2*x[i2-1] - 2*x[i2])/Δx^2
    end

    nothing 
end


# 1D heat equation
function heat_eq!(dx,x,p,t)       
    # time = t/Tf;
    #u = input_signal(time, p)
    
    u_in = input_signal(t, u_raw)
    
    diffusion_x!(dx,x,Nx,1,Δx)
  
    dx .= α * dx
    dx[1] = dx[1] + 2α/(λ * Δx) * u_in
end


# Discretization  
const Nx = 101;     # Number of elements x-direction
const Δx = L/(Nx-1) # Spatial sampling
#const Tf = 1000.0;  # Final time
const ts = Tf/60   # Time step width

# Simulation without optimization
using OrdinaryDiffEq

x0 = 300 * ones(Nx) # Intial values
tspan = (0.0, Tf)   # Time span
alg = KenCarp4()    # Numerical integrator

prob = ODEProblem(heat_eq!,x0,tspan)
sol = solve(prob,alg, saveat = ts)

using CairoMakie

path2folder = "results/figures/controlled/"
filename = path2folder*"gevrey_"*string(round(Int64,Tf))*".pdf"
begin
    input_data = mapreduce(t->input_signal(t,u_raw),vcat,sol.t);
    input_data = Float64.(input_data / 1e5)
    sol_data = sol[1:50:101,:]'
    
    f = Figure(size=(1300,400),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel = "Time t in [s]", ylabel = L"Input $\times 10^5$", 
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    ax1.xticks = 0 : Tf/4 : Tf;
    #ax1.yticks = 0 : 2.5 : 10;
    ax1.yticks = -10 : 5 : 20;
    lines!(sol.t, input_data; linestyle = :dash, linewidth = 5)


    ax2 = Axis(f[1, 2], xlabel = "Time t in [s]", ylabel = "Temperature in [K]", 
                xlabelsize = 30,  ylabelsize = 30,
                xgridstyle = :dash, ygridstyle = :dash,)

    ax2.xticks = 0 : Tf/4 : Tf;
    #ax2.yticks = 300 : 20 : 460;
    lines!(sol.t, sol_data[:,1]; linestyle = :dot, linewidth = 5, label = "Left")
    lines!(sol.t, sol_data[:,2]; linestyle = :dash,linewidth = 5, label = "Center")
    lines!(sol.t, sol_data[:,3];                   linewidth = 5, label = "Right")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f  
    save(filename, f, pt_per_unit = 1)   
end



#=
input_data = mapreduce(t->input_signal(t,u_raw),vcat,sol.t);
input_data = Float64.(input_data / maximum(abs.(input_data)))
input_data = round.(input_data,digits=3)
input_csv = hcat(sol.t,input_data)
sol_data = sol[1:50:101,:]'
sol_data =  round.(sol_data/ maximum(sol_data), digits=3) 
sol_csv = hcat(sol.t,sol_data)


using DelimitedFiles;
path2folder = "results/data/"
filename = "gevrey_input_"*string(round(Int64,Tf))*".csv"
path2file = path2folder * filename

open(path2file, "w") do io
    writedlm(io, ["t" "u"], ',')
    writedlm(io, input_csv, ',')
end;

filename = "gevrey_temp_"*string(round(Int64,Tf))*".csv"
path2file = path2folder * filename

open(path2file, "w") do io
    writedlm(io, ["t" "x" "y" "z"], ',')
    writedlm(io, sol_csv, ',')
end;
=#