
using BellBruno
N_der = 10;              # Number of derivatives

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
Tf = 1200;    # Final simulation time
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

#=
diff_ref = 100; # (y_f - y_0) = 100 Kelvin
ref_data = ref_init .+ diff_ref*ref.(tgrid1)
ref_raw = hcat(ref_data,(diff_ref*h_data)/ω_int)
=#

# ref_init = 300; # Intial Temperature



L = 0.1; # Length of 1D rod
W = 0.05
N₁ = 3;
N₂ = 5;
Nc = N₁*N₂ 
Δx₁ = L/N₁
Δx₂ = W/N₂

# Aluminium
λ₁ = 40;  # Thermal conductivity
λ₂ = 60;
ρ = 8000; # Density
c = 400;  # Specific heat capacity
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
# B = vcat(diagm(ones(Int64,N₁)), zeros(Int64,N₁*(N₂-1),N₁)) / b1

# Scenario: 
# 1 = faulty actuators / with spatial characteristics  
# 2 = faulty sensors / with spatial characteristics
sc = 2

if sc==1
    Bd = diagm([1,0.9,0.8])/ b1;
    Cd = diagm(ones(3))
else
    Bd = diagm(ones(3))/ b1 
    Cd = diagm([1,0.9,0.8])
end

B = vcat(Bd, zeros(Int64,N₁*(N₂-1),N₁)) 

C = hcat(zeros(Int64,N₁,N₁*(N₂-1)),Cd);

# Original

Om = mapreduce(i-> C*A^i,vcat,0:N₂-1)
Om_inv = inv(Om)

Mu1 = inv(C*A^(N₂-1)*B)
Mu = hcat(-Mu1*C*A^N₂ * Om_inv, Mu1)
sum(Mu, dims=1)

Θinit = 0
Θdes = 100
ref_init = Θinit*Cd*ones(3) # [310,300,320]

dref1 = Θdes-ref_init[1] # 90  # (y_f - y_0) = 100 Kelvin
dref2 = Θdes-ref_init[2] # 100 # (y_f - y_0) = 100 Kelvin
dref3 = Θdes-ref_init[3] # 80  # (y_f - y_0) = 100 Kelvin
ref01 = ref_init[1] .+ dref1*ref.(tgrid1)
ref02 = ref_init[2] .+ dref2*ref.(tgrid1)
ref03 = ref_init[3] .+ dref3*ref.(tgrid1)

dref_raw =mapreduce(i-> hcat(dref1*h_data[:,i],dref2*h_data[:,i],dref3*h_data[:,i])/ω_int, hcat,1:N_der+1)
ref_mimo = hcat(ref01, ref02, ref03, dref_raw)


u_raw = (Mu*ref_mimo[:,1:Nc+N₁]')'

# using Plots
# plot(u_raw)


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
    
    u_in1 = input_signal(t, u_raw[:,1])
    u_in2 = input_signal(t, u_raw[:,2])
    u_in3 = input_signal(t, u_raw[:,3])
    
    u_in = [u_in1, u_in2, u_in3]
    dx .= A*x + B*u_in # B[:,1]*1 + B[:,2]*2 + B[:,3]*3 #*u_in
end

const Nts = 40;
const ts = Tf/Nts   # Time step width

# Simulation without optimization
using OrdinaryDiffEq

x0 = Θinit * ones(Nc) # repeat(ref_init,N₂) # 300 * ones(Nc) # Intial values
tspan = (0.0, Tf)   # Time span
alg = KenCarp4()    # Numerical integrator

prob = ODEProblem(heat_eq!,x0,tspan)
sol = solve(prob,alg, saveat = ts)

#=
plot(sol[N₁*(N₂-1)+1:Nc,:]')
yout = C*Array(sol)
plot(sol.t, yout')
plot(sol)
=#
yout = C*Array(sol)

using CairoMakie

path2folder = "results/figures/controlled/"
filename = path2folder*"fbc_2d_approx_scenario_"*string(sc)*".pdf"

begin
    input_data = zeros(Nts, N₁)
    input_data = mapreduce(t->hcat(input_signal(t,u_raw[:,1]),input_signal(t,u_raw[:,2]),input_signal(t,u_raw[:,3])),vcat,sol.t);
    input_data = Float64.(input_data / 1e4)
    # sol_data = hcat(sol[1,:],sol[N₁*(round(Int64,N₂/2)-1)+1,:],sol[N₁*(N₂-1)+1,:]) # hcat(sol[1,:],sol[2N₁+1,:],sol[4N₁+1,:])
    sol_data = sol[N₁*(N₂-1)+1:Nc,:]'
    

    # f = Figure(size=(1300,400),fontsize=26)
    f = Figure(size=(600,900),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel = "", ylabel = L"Input $\times 10^4$", 
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    ax1.xticks = 0 : Tf/4 : Tf;
    ax1.yticks = -4 : 2 : 12;
    #ax1.yticks = -10 : 5 : 20;
    lines!(sol.t, input_data[:,1]; linestyle = :solid, linewidth = 5 , label = "Input 1")
    lines!(sol.t, input_data[:,2]; linestyle = :dash, linewidth = 5, label = "Input 2")
    lines!(sol.t, input_data[:,3]; linestyle = :dot, linewidth = 5, label = "Input 3")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);


    ax2 = Axis(f[2, 1], xlabel = "Time t in [s]", ylabel = "Temperature", 
                xlabelsize = 30,  ylabelsize = 30,
                xgridstyle = :dash, ygridstyle = :dash,)

    ax2.xticks = 0 : Tf/4 : Tf;
    ax2.yticks = 0 : 25 : 125;
    lines!(sol.t, sol_data[:,1]; linestyle = :solid, linewidth = 5, label = L"$i=N_{c}-2$")
    lines!(sol.t, sol_data[:,2]; linestyle = :dash,  linewidth = 5, label = L"$i=N_{c}-1$")
    lines!(sol.t, sol_data[:,3]; linestyle = :dot,   linewidth = 5, label = L"$i=N_{c}$")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f  
    save(filename, f, pt_per_unit = 1)   
end

