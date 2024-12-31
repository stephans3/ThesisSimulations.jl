
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
Tf = 10000;    # Final simulation time
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

#=
diff_ref = 100; # (y_f - y_0) = 100 Kelvin
ref_data = ref_init .+ diff_ref*ref.(tgrid1)
ref_raw = hcat(ref_data,(diff_ref*h_data)/ω_int)
=#


dref1 = 100; # (y_f - y_0) = 100 Kelvin
dref2 = 200; # (y_f - y_0) = 100 Kelvin
dref3 = 80; # (y_f - y_0) = 100 Kelvin
ref01 = ref_init .+ dref1*ref.(tgrid1)
ref02 = ref_init .+ dref2*ref.(tgrid1)
ref03 = ref_init .+ dref3*ref.(tgrid1)

dref_raw =mapreduce(i-> hcat(dref1*h_data[:,i],dref2*h_data[:,i],dref3*h_data[:,i])/ω_int, hcat,1:N_der+1)
ref_raw = hcat(ref01, ref02, ref03, dref_raw)


L = 0.3; # Length of 1D rod

###### ACHTUNG
W = 0.02
N₂ = 5;
###### ACHTUNG


# Aluminium
λ₁ = 40;  # Thermal conductivity
λ₂ = 60;
ρ = 8000; # Density
c = 400;  # Specific heat capacity
α₁ = λ₁ / (ρ * c) # Diffusivity
α₂ = λ₂ / (ρ * c) # Diffusivity

N₁ = 3;
Nc = N₁*N₂ 
Δx₁ = L/N₁
Δx₂ = W/N₂


# -u[1] + u[2]
# u[i-1] - 2u[i] + u[i]
# u[N₁-1] - u[N₁]

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

D1

#D2[1+N₁,1:N₁:1+2N₁]
1+(N₂-1)*N₁
for i=1:N₁
    for j=2:N₂-1
        di = (j-1)*N₁
        D2[i+di,i+di-N₁:N₁:(i+di)+N₁] =  [1,-2,1]
        #D2[i+(j-1)*N₁,i-1+(j-1)*N₁ : i+1+(j-1)*N₁] = [1,-2,1];
    end
    D2[i,i:N₁:i+N₁] = [-1,1]
    D2[i+(N₂-1)*N₁,i+(N₂-2)*N₁:N₁:i+(N₂-1)*N₁] = [1,-1]
end

a1 = 1 # α₁/Δx₁;
a2 = 1 # α₂/Δx₂;

# D = D1 + D2

using LinearAlgebra
A = a1*D1 + a2*D2
b1 = (Δx₂*c*ρ)
B = vcat(diagm(ones(Int64,N₁)), zeros(Int64,N₁*(N₂-1),N₁)) / b1
C = hcat(zeros(Int64,N₁,N₁*(N₂-1)),diagm(ones(Int64,N₁)));

C*A^(Nc)*B

# Original

Om = mapreduce(i-> C*A^i,vcat,0:N₂-1)
Om_inv = inv(Om)

Mu1 = inv(C*A^(N₂-1)*B)
Mu = hcat(-Mu1*C*A^N₂ * Om_inv, Mu1)
sum(Mu, dims=1)


# Modified
Md1 = diagm(mapreduce(i-> a1^i*ones(N₁),vcat,0:N₂-1))
Om1 = mapreduce(i-> C*(D1+(a2/a1)*D2)^i,vcat,0:N₂-1)
# Omm = Md1*Om1
Om_inv = inv(Om1)*inv(Md1)

Mu11 = B'*B*C*C' * (a1/a2)^(N₂-1) # inv(C*(D1+(a2/a1)*D2)^(N₂-1)*B) #/ a1^((N₂-1))
Mu12 = C*(D1+(a2/a1)*D2)^N₂
Mu13 = B'*B*C*C' * (1/a2)^(N₂-1)
Mu_new = hcat(-a1*Mu11*Mu12*Om_inv, Mu13)

# scatter((Mu-Mu_new)[1,1:15])
# err1 = sum(Mu, dims=1) - sum(Mu_new, dims=1)
# scatter(log10.(abs.(err1[1,1:end])))

ref_mimo = ref_raw # = mapreduce(i->repeat(ref_raw[:,i],1,N₁), hcat, 1:N₂+1)'
u_raw = (Mu*ref_mimo[:,1:Nc+N₁]')'
#u_raw2 = (Mu_new*ref_mimo)'


using Plots
plot(u_raw[:,1])

plot(u_raw2[:,1])

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


const ts = Tf/60   # Time step width

# Simulation without optimization
using OrdinaryDiffEq

x0 = 300 * ones(Nc) # Intial values
tspan = (0.0, Tf)   # Time span
alg = KenCarp4()    # Numerical integrator

prob = ODEProblem(heat_eq!,x0,tspan)
sol = solve(prob,alg, saveat = ts)

using Plots
plot(sol)

plot(sol[end-2:end,:]')

dx0 = similar(x0)
heat_eq!(dx0,x0,0,0)       