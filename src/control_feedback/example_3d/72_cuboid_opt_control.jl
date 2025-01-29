#=
    Reference tracking optimization of 3-dimensional heat conduction
=#

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
setEmission!(boundary, emission, :topside )

### Actuation ###
actuation = IOSetup(cuboid)
num_actuators = (2,2)        # Number of actuators per boundary
config  = RadialCharacteristics(1.0, 2, 30)
# config  = RadialCharacteristics(1.0, 2, 0)
setIOSetup!(actuation, cuboid, num_actuators, config, :underside)


# Reference
ref_init = 300;
Δr = 200; # Difference operating points
ps = 10; # Steepness
ψ(t) = (1+tanh(10*(t/Tf - 1/2)))/2
ref(t) = ref_init + Δr*ψ(t)


function input_obc(t,p)
    return exp(p[1] - p[3]^2 * (t/Tf - 1/p[2])^2)
end



function heat_conduction!(dθ, θ, param, t)
    u1 = input_obc(t,param[1:3])
    u2 = input_obc(t,param[4:6])
    u3 = input_obc(t,param[7:9])

    u_in = [u1, u2, u3, u1]
    diffusion!(dθ, θ, cuboid, property, boundary, actuation, u_in)
end


#=
p_energy = [13.065247340633551
             2.070393374741201
             9.076210596231089]
=#

p_energy = [13.065247340633551
             2.070393374741201
             9.076210596231089]


pinit = repeat(p_energy,4)
Tf = 1200;
tspan = (0.0, Tf)
θ₀ = 300;
θinit = θ₀*ones(Nc)
dΘ = similar(θinit)
heat_conduction!(dΘ,θinit,pinit,0)

using OrdinaryDiffEq
t_samp = 30.0
alg = KenCarp5()
prob = ODEProblem(heat_conduction!,θinit,tspan,pinit)
sol = solve(prob,alg,p=pinit, saveat=t_samp)



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

ref_data = repeat(ref.(sol.t)',Ny)
y = C*Array(sol)
err = ref_data - y

# loss = sum(abs2, err) / Tf
loss = sum(abs2, err)*t_samp / Tf


function loss_optim(u,p)
    pars = u
    sol_loss = solve(prob, alg,p=pars, saveat = t_samp)
    if sol_loss.retcode == ReturnCode.Success
        y = C*Array(sol_loss)
        err = ref_data - y
        loss = sum(abs2, err) *t_samp / Tf
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


    println(l)
    #println("iter")

    if l < -0.2
        return true
    end

    return false
end


pinit = repeat(p_energy,3)
loss_optim(pinit, [0])

#=
    using Optimization, OptimizationOptimJL, ForwardDiff
    optf = OptimizationFunction(loss_optim, Optimization.AutoForwardDiff())
    opt_prob = OptimizationProblem(optf,pinit, [0])
    p_opt = Optimization.solve(opt_prob, ConjugateGradient(), callback=callback, maxiters=60)
=#

p_opt=
[12.894929657082693
  2.0554514321527058
  7.682015031653755
 12.967392692355489
  2.0538122351888752
  8.300509555164869
 12.943662543893003
  2.066455923317032
  8.459445553931738]


loss_optim(p_opt, [0])
sol_oc = solve(prob,alg,p=p_opt, saveat=t_samp)
y_oc = C*Array(sol_oc)

using CairoMakie
path2folder = "results/figures/controlled/cuboid_example/"
begin
    filename = path2folder*"cuboid_opt_control_output_1.pdf"

    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Time in [s]", ylabel = L"Output in [K] $~$", xlabelsize = 30, ylabelsize = 30,
    xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,) #, limits = (nothing, (499.2, 500.05))
    
    tgrid = sol_oc.t[1:end]
    lines!(tgrid, y_oc[1,:], linestyle = :solid,  linewidth = 5, label="Sensor 1")
    lines!(tgrid, y_oc[2,:], linestyle = :dash,  linewidth = 5, label="Sensor 2")
    lines!(tgrid, y_oc[3,:], linestyle = :dot,  linewidth = 5, label="Sensor 3")
    lines!(tgrid, y_oc[4,:], linestyle = :dashdot,  linewidth = 5, label="Sensor 4")
    scatter!(tgrid[1:5:end], ref_data[1,1:5:end];  markersize = 15, marker = :diamond, color=:black, label = "Reference")
    ax1.xticks = collect(0:200:1200)
#    ax1.yticks = [499.2, 499.4, 499.6, 499.8, 500];
    axislegend(; position = :rb, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f
    
    save(filename, f, pt_per_unit = 1)   
end

begin

    filename = path2folder*"cuboid_opt_control_output_2.pdf"

    f = Figure(size=(600,400),fontsize=26)
    ax1 = Axis(f[1, 1], xlabel = "Time in [s]", ylabel = L"Output in [K] $~$", xlabelsize = 30, ylabelsize = 30,
    xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1)#, limits = (nothing, (490, 510)))
    
    tstart = 26;
    tgrid = sol_oc.t[tstart:end]
    lines!(tgrid, y_oc[1,tstart:end], linestyle = :solid,  linewidth = 5, label="Sensor 1")
    lines!(tgrid, y_oc[2,tstart:end], linestyle = :dash,  linewidth = 5, label="Sensor 2")
    lines!(tgrid, y_oc[3,tstart:end], linestyle = :dot,  linewidth = 5, label="Sensor 3")
    lines!(tgrid, y_oc[4,tstart:end], linestyle = :dashdot,  linewidth = 5, label="Sensor 4")
    scatter!(sol_oc.t[26:end], ref_data[1,26:end];  markersize = 15, marker = :diamond, color=:black, label = "Reference")

    ax1.xticks = collect(0:100:1200)
#    ax1.yticks = [499.2, 499.4, 499.6, 499.8, 500];
    #axislegend(; position = :rb, backgroundcolor = (:grey90, 0.1), labelsize=30);
    f
    
    save(filename, f, pt_per_unit = 1)   
end


#=
Workstation run:
p_opt=
[12.894929657082693
  2.0554514321527058
  7.682015031653755
 12.967392692355489
  2.0538122351888752
  8.300509555164869
 12.943662543893003
  2.066455923317032
  8.459445553931738]

st_pars = hcat(store_param...)'
st_pars =
[13.0652  2.07039  9.07621  13.0652  2.07039  9.07621  13.0652  2.07039  9.07621
 13.0236  2.05769  9.08007  13.046   2.06325  9.07799  13.0428  2.06486  9.0783
 13.0344  2.06139  9.07845  13.0524  2.0642   9.07705  13.0464  2.06703  9.07766
 13.0325  2.0608   9.07833  13.052   2.06338  9.07692  13.0444  2.06701  9.07771
 13.032   2.06064  9.0781   13.0522  2.06286  9.07675  13.0431  2.06714  9.0777
 13.0316  2.06056  9.07762  13.0529  2.06203  9.0764   13.0412  2.06746  9.07763
 13.0316  2.06076  9.0758   13.0557  2.05944  9.07513  13.0354  2.06875  9.0773
 13.0351  2.06222  9.07211  13.062   2.05575  9.07268  13.0273  2.07131  9.07645
 13.0351  2.06235  9.07101  13.0626  2.055    9.07201  13.0253  2.07157  9.07612
 13.035   2.06237  9.07051  13.0625  2.05485  9.07174  13.0248  2.07155  9.07594
 13.035   2.06241  9.06987  13.0624  2.05472  9.07139  13.0241  2.07149  9.0757
 13.035   2.06247  9.06908  13.0622  2.0546   9.07096  13.0235  2.07141  9.0754
 13.0352  2.06267  9.06711  13.0617  2.05439  9.06991  13.0221  2.07121  9.07463
 13.037   2.06388  9.05659  13.0595  2.05348  9.0643   13.0154  2.07016  9.07048
 13.0488  2.06915  9.01567  13.0525  2.05122  9.04259  12.9944  2.06662  9.05409
 13.0476  2.0683   9.00793  13.0492  2.05036  9.0385   12.9907  2.06561  9.05085
 13.0465  2.06515  8.99786  13.0447  2.0503   9.03328  12.9939  2.0655   9.04597
 13.0454  2.05811  8.96902  13.0338  2.05184  9.01832  13.007   2.06685  9.03177
 13.0331  2.06339  8.93175  13.0228  2.06094  8.99826  13.0202  2.07494  9.01343
 13.0328  2.05625  8.91714  13.0316  2.05903  8.98948  13.02    2.0722   9.00694
 13.0282  2.0537   8.90933  13.0361  2.05797  8.98459  13.0166  2.07085  9.00361
 13.0141  2.06066  8.87478  13.0617  2.05905  8.96282  13.0019  2.07141  8.9891
 13.0163  2.06105  8.87302  13.0614  2.0586   8.96199  13.0041  2.07153  8.9882
 13.0167  2.05967  8.86799  13.053   2.05528  8.95998  13.0099  2.0707   8.98545
 13.0162  2.06063  8.86403  13.0462  2.05411  8.95836  13.0132  2.07063  8.98335
 13.0178  2.06259  8.85998  13.0419  2.05399  8.95658  13.0164  2.07086  8.98131
 13.0173  2.06246  8.85896  13.0412  2.05396  8.95605  13.016   2.07055  8.98086
 13.0169  2.06226  8.85775  13.0409  2.05403  8.95538  13.0154  2.07018  8.98037
 13.0166  2.06173  8.8532   13.0406  2.0545   8.95286  13.0131  2.06891  8.97853
 13.0179  2.05998  8.82694  13.0412  2.05773  8.93826  13.0017  2.06255  8.96788
 13.0211  2.06143  8.81547  13.0425  2.05935  8.93193  12.999   2.06191  8.96313
 13.0174  2.06327  8.80408  13.0395  2.05879  8.92571  12.9959  2.06443  8.95819
 13.0163  2.06757  8.7782   13.0358  2.05635  8.91169  12.9951  2.07229  8.9467
 13.0104  2.06347  8.71755  13.0255  2.04521  8.87883  12.9952  2.08417  8.91949
 13.0111  2.05895  8.68498  13.0235  2.04222  8.86107  12.9973  2.08553  8.90487
 13.0053  2.05258  8.64305  13.0193  2.04211  8.83799  12.9968  2.08264  8.8861
 12.9831  2.04215  8.47517  13.0075  2.05563  8.7452   12.9941  2.06859  8.81113
 12.9661  2.04174  8.29055  12.9997  2.0745   8.64309  12.9938  2.05738  8.72876
 12.9347  2.05352  8.11258  12.988   2.08905  8.54424  12.9828  2.05564  8.64973
 12.9321  2.05642  8.03408  12.99    2.08215  8.50055  12.9794  2.06083  8.61519
 12.907   2.05136  7.85397  12.9859  2.05193  8.4      12.9595  2.07202  8.53622
 12.9048  2.05237  7.81879  12.9859  2.04712  8.3804   12.9567  2.07431  8.52081
 12.9007  2.05998  7.75937  12.982   2.04572  8.3474   12.9512  2.07514  8.49478
 12.8948  2.0611   7.72944  12.9775  2.04489  8.33075  12.9466  2.07343  8.48165
 12.8946  2.05796  7.70874  12.9748  2.0456   8.31928  12.9457  2.06891  8.47249
 12.8956  2.05702  7.70342  12.9741  2.04669  8.31634  12.9462  2.06743  8.4701
 12.8958  2.05587  7.69802  12.9726  2.04856  8.31332  12.9465  2.06568  8.46764
 12.8947  2.05532  7.69471  12.9707  2.05027  8.31143  12.9462  2.06477  8.4661
 12.8947  2.05573  7.69182  12.9694  2.05227  8.30975  12.9464  2.06473  8.46474
 12.8948  2.05579  7.69162  12.9693  2.05237  8.30962  12.9465  2.06488  8.46464
 12.8949  2.05581  7.69153  12.9693  2.0524   8.30956  12.9465  2.06498  8.46459
 12.8952  2.05584  7.6909   12.969   2.05249  8.30908  12.9466  2.06587  8.46426
 12.8949  2.05529  7.68935  12.9679  2.05239  8.30782  12.9463  2.06802  8.46343
 12.8951  2.0548   7.68882  12.9678  2.05229  8.30736  12.9462  2.06823  8.46315
 12.8952  2.05461  7.6886   12.9678  2.05228  8.30715  12.9462  2.06818  8.46303
 12.8952  2.05449  7.68842  12.9678  2.05229  8.30699  12.9461  2.0681   8.46294
 12.8952  2.05437  7.6882   12.9678  2.05232  8.30676  12.946   2.06798  8.46282
 12.8953  2.05412  7.68734  12.9678  2.05253  8.3059   12.9456  2.06749  8.46237
 12.8953  2.0541   7.68594  12.9679  2.05296  8.30449  12.945   2.06686  8.46162
 12.8953  2.05448  7.68478  12.9678  2.05329  8.30332  12.9446  2.06667  8.46099
 12.8949  2.05545  7.68202  12.9674  2.05381  8.30051  12.9437  2.06646  8.45945]

store_loss=
[
    118.36387253849452
  57.98815397619244
  54.31508070033389
  53.84548008566462
  53.724157313502495
  53.567453685371156
  53.13404987732049
  52.51977629673134
  52.46983037600543
  52.45187835058337
  52.43246825644163
  52.409222358842726
  52.3506554091587
  52.07183991604226
  51.73306960275746
  51.386789101719216
  50.74564971049809
  49.76066906555915
  48.91927941205708
  47.78871888884504
  47.285076125915836
  46.28034953537747
  46.040235280683845
  45.26885174238742
  45.037318691373144
  44.925047552225614
  44.87792346621077
  44.84196153509327
  44.71823387063663
  44.16274152456638
  43.83136390656821
 43.30626859575459
 42.63640316815894
 41.25560334131272
 40.57630194757537
 39.51866028332606
 36.26022429247911
 32.297097198086526
 29.155142633698567
 27.16244095215276
 21.017760365194473
 20.7074014983112
 20.406407145342563
 19.866480209069167
 19.326065620132614
 19.21897699067677
 19.137562914324594
 19.06648917959118
 19.016930484620005
 19.013360704251166
 19.012072782397407
 19.007311787887126
 18.98465542022628
 18.978154609518505
 18.975812945169814
 18.974170175086126
 18.9719895598673
 18.96457301772109
 18.954701295567208
 18.944912175273
 18.931369535689146
]

=#










