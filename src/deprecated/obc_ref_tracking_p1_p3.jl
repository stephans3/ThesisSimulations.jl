using Hestia
prop = StaticIsotropic(50, 8000, 400)

L = 0.1
N = 10
rod = HeatRod(L, N)

boundary = Boundary(rod)
h = 10;
Θamb = 300;
ε = 0.2;
em_total = Emission(h, Θamb,ε) 
#setEmission!(boundary, em_total, :west);
setEmission!(boundary, em_total, :east);

actuation = IOSetup(rod)
setIOSetup!(actuation, rod, 1, 1.0,  :west)

function input_obc(t,p)
    return exp(p[1] - p[3]^2 * (t/Tf - 1/p[2])^2)
end


function heat_conduction!(dw, w, param, t) 
    u_in1 = ones(1)*input_obc(t,param)

    diffusion!(dw, w, rod, prop, boundary,actuation,u_in1)
end


p1 = 11.22206881384893;
p2 = 2.1413276231263385;
p3_vals=[
 12.263851926618486
 12.32242367592184
 12.437396186816946
]

using OrdinaryDiffEq
θinit =  300* ones(N) # Intial values
Tf    =  3000;
tspan =  (0.0, Tf)   # Time span
tsamp = 1.0;
p_orig= [p1,p2,p3_vals[2]]
alg = KenCarp4()    # Numerical integrator
prob_orig = ODEProblem(heat_conduction!,θinit,tspan,p_orig)
sol_orig = solve(prob_orig, alg, saveat = tsamp)


using FastGaussQuadrature
p_cntr = 2.0
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

tgrid = 0 : tsamp : Tf
ref_init = 300; # Intial Temperature
dref = 100; # (y_f - y_0) = 100 Kelvin
ref_data = ref_init .+ dref*ref.(sol_orig.t)

# err1 = ref_data - sol_orig[end,:]
# loss1 = sum(abs2,err1) / Tf

function loss_optim_std(u,p)
    pars = [u[1], p[1], u[2]]
    sol1 = solve(prob_orig, alg,p=pars, saveat = tsamp)
    if sol1.retcode == ReturnCode.Success
        y = sol1[end,:]
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

    #println("iter")

    return false
end


# Find optimal p3
using Optimization, OptimizationOptimJL, ForwardDiff
opt_p = [p2]
opt_u0 = [p1, p3_vals[2]]
loss_optim_std(opt_u0, opt_p)

optf = OptimizationFunction(loss_optim_std, Optimization.AutoForwardDiff())
opt_prob = OptimizationProblem(optf, opt_u0, opt_p)
p1p3_opt = Optimization.solve(opt_prob, ConjugateGradient(), callback=callback)

p13_path = hcat(store_param...)'
p1grid = 11.2:0.005:11.3;
p2grid = 12:0.01:12.4
loss2d = zeros(length(p1grid),length(p2grid))

for (i1, p11) in enumerate(p1grid), (i2,p22) in enumerate(p2grid)
    loss2d[i1,i2] = loss_optim_std([p11,p22],[p3_vals[2]])
end

using Plots
plot(p12_path[:,1])
contourf(p2grid,p1grid,loss2d)