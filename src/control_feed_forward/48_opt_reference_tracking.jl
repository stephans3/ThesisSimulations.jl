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
setEmission!(boundary, em_total, :west);
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
    pars = [u[1], u[2], p[1]]
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
opt_p = [p3_vals[2]]
opt_u0 = [p1, p2]
loss_optim_std(opt_u0, opt_p)

optf = OptimizationFunction(loss_optim_std, Optimization.AutoForwardDiff())
opt_prob = OptimizationProblem(optf, opt_u0, opt_p)
p12 = Optimization.solve(opt_prob, ConjugateGradient(), callback=callback, maxiters=100)

p12_path = hcat(store_param...)'
p1grid = 11.21:0.005:11.35;
p2grid = 2.05:0.005:2.2
loss2d = zeros(length(p1grid),length(p2grid))

for (i1, p11) in enumerate(p1grid), (i2,p22) in enumerate(p2grid)
    loss2d[i1,i2] = loss_optim_std([p11,p22],[p3_vals[2]])
end


input1(t) = input_obc(t,[p1, p2,p3_vals[2]])
input2(t) = input_obc(t,vcat(p12,p3_vals[2]))

input1_data = input1.(sol_orig.t)
input2_data = input2.(sol_orig.t)

pars = vcat(p12,p3_vals[2])
sol_adjust = solve(prob_orig, alg,p=pars, saveat = tsamp)

using CairoMakie
path2folder = "results/figures/controlled/"
filename = path2folder*"optim_based_1d_contour_param_fit.pdf"
begin
    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = L"Gain $p_{1}$", ylabel = L"Time shift $p_{2}$", xlabelsize = 30, ylabelsize = 30)
    #tightlimits!(ax1)
    #hidedecorations!(ax1)
    # co = contourf!(ax1, p1grid, p2grid, loss2d, levels=20, colormap=:Greens) #levels = range(0.0, 10.0, length = 20))
    co = contourf!(ax1, p1grid, p2grid, loss2d, levels=20, colormap=:managua) #levels = range(0.0, 10.0, length = 20))
    #lines!(p12_path[1:20,1],p12_path[1:20,2], linestyle = :dash,  linewidth = 5, color=:black)#RGBf(0.5, 0.2, 0.8))
    lines!(p12_path[1:end,1],p12_path[1:end,2], linestyle = :dash,  linewidth = 5, color=:black)#RGBf(0.5, 0.2, 0.8))
    scatter!([p12[1]],[p12[2]], marker = :xcross, markersize=25, color=:purple)
    ax1.xticks = 11.22 : 0.04 : 11.34 #[11.2, 11.24, 11.28, 11.32, 11.36];
    ax1.yticks = 2.05:0.05:2.2;
    Colorbar(f[1, 2], co)
    f    

    save(filename, f, pt_per_unit = 1)   
end


filename = path2folder*"optim_based_1d_input_param_fit.pdf"
begin   
    f = Figure(size=(600,400),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel = "Time t in [s]", ylabel=L"Input $\times 10^4$",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    scale = 1e-4;

    ax1.xticks = 0 : 500 : 3000;
    ax1.yticks = 0 : 2.5 : 10;

    tstart = 450;
    tend = 2550;
    lines!(sol_orig.t, scale*input1_data;   linestyle = :dot,   linewidth = 5, label="Initial")
    lines!(sol_orig.t, scale*input2_data;   linestyle = :dash,   linewidth = 5, label="Adjusted")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);
      
    f  
    save(filename, f, pt_per_unit = 1)   
end

filename = path2folder*"optim_based_1d_output_param_fit.pdf"
begin   
    y_data_orig = sol_orig[end,:]
    y_data_adjust = sol_adjust[end,:]
    
    f = Figure(size=(600,400),fontsize=26)

    ax1 = Axis(f[1, 1], xlabel = "Time t in [s]", ylabel="Temperature in [K]",
    xlabelsize = 30,  ylabelsize = 30,
    xgridstyle = :dash, ygridstyle = :dash,)

    scale = 1e-4;

    ax1.xticks = 0 : 500 : 3000;
    ax1.yticks = 300 : 25 : 400;

    tstart = 450;
    tend = 2550;
    lines!(sol_orig.t, y_data_orig;   linestyle = :dot,   linewidth = 5, label="Initial")
    lines!(sol_orig.t, y_data_adjust;   linestyle = :dash,   linewidth = 5, label="Adjusted")
    scatter!(sol_orig.t[1:250:end], ref_data[1:250:end];  markersize = 15, marker = :diamond, color=:black, label = "Reference")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1), labelsize=30);
      
    ax2 = Axis(f, bbox=BBox(411, 569, 170, 280), ylabelsize = 24)
    ax2.xticks = 2000 : 500 : Tf;
    ax2.yticks = [390, 395,400];
    lines!(ax2, sol_orig.t[2000:end],y_data_orig[2000:end];   linestyle = :dot,  linewidth = 5, color=Makie.wong_colors()[1])
    lines!(ax2, sol_orig.t[2000:end],y_data_adjust[2000:end]; linestyle = :dash, linewidth = 5, color=Makie.wong_colors()[2])
    translate!(ax2.scene, 0, 0, 10);

    f  
    save(filename, f, pt_per_unit = 1)   
end




