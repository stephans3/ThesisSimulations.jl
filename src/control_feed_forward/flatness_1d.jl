

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
Tf = 3000;    # Final simulation time
N_dt = 1000;
dt = Tf/N_dt; # Sampling time
h_data = compute_derivatives(N_der, bc, bp, Tf, dt, w=p_cntr);



using FastGaussQuadrature
bump(t) = exp(-1 / (t/Tf - (t/Tf)^2)^p_cntr)
t_gq, weights_gq = FastGaussQuadrature.gausslegendre(1000)
tshift = Tf/2;
ω_int = tshift *FastGaussQuadrature.dot( weights_gq ,bump.(tshift*t_gq .+ tshift))


function ref(t)
    ts1 = t/2
    if t <= 0
        # println("1")
        return 0
    elseif t >= Tf
        # println("2")
        return 1
    else
        # println("3")
        return ts1*FastGaussQuadrature.dot( weights_gq ,bump.(ts1*t_gq .+ ts1))/ω_int;
    end
end

tgrid1 = dt : dt : Tf-dt; # Time grid
ref_init = 300; # Intial Temperature
diff_ref = 100; # (y_f - y_0) = 100 Kelvin
ref_data = diff_ref*ref.(tgrid1)
ref_raw = hcat(ref_data,(diff_ref*h_data)/ω_int)

##################
#=
- Compute input signal u(t) for aluminum and steel 38Si7
=#

L = 0.1; # Length of 1D rod
# Aluminium
λ = 50;  # Thermal conductivity
ρ = 8000; # Density
c = 400;  # Specific heat capacity
α = λ / (ρ * c) # Diffusivity

Nx = 20;
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
C = hcat(zeros(Int64,1,Nx-1),1)

Om = mapreduce(i-> C*D1^i,vcat,0:Nx-1)
Om_inv = inv(Om)
Om_inv = (Om_inv' .* mapreduce(i-> a1_inv^i,vcat,0:Nx-1))'
Mu = hcat((a1*b1)*(-C*D1^Nx*Om_inv),b1*a1_inv^(Nx-1))
u_raw = (Mu*ref_raw[:,1:Nx+1]')'


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

# 1D heat equation
function heat_eq!(dx,x,p,t)       
    # time = t/Tf;
    #u = input_signal(time, p)
    
    u_in = input_signal(t, u_raw)
    
    dx .= A*x + B*u_in
end


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
filename = path2folder*"fbc_ode_"*string(round(Int64,Tf))*".pdf"
begin
    input_data = mapreduce(t->input_signal(t,u_raw),vcat,sol.t);
    input_data = Float64.(input_data / 1e4)
    # sol_data = hcat(sol[1,:],sol[3,:],sol[5,:])
    sol_data = hcat(sol[1,:],sol[10,:],sol[20,:])
    
    f = Figure(size=(1300,400),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel = "Time t in [s]", ylabel = L"Input $\times 10^4$", 
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    ax1.xticks = 0 : Tf/4 : Tf;
    ax1.yticks = 0 : 2.5 : 10;
    #ax1.yticks = -10 : 5 : 20;
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
