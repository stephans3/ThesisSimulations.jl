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


p_opt = [13.014169886578932
2.067130614022539
8.782250563794914
13.035559079377817
2.0539846802239246
8.913880017111973
12.996417090042813
2.071839932635213
8.948344730366804]

# repeat(p_energy,3)


loss_optim(p_opt, [0])

using Optimization, OptimizationOptimJL, ForwardDiff
optf = OptimizationFunction(loss_optim, Optimization.AutoForwardDiff())
opt_prob = OptimizationProblem(optf, p_opt, [0])
p_opt = Optimization.solve(opt_prob, ConjugateGradient(), callback=callback, maxiters=60)


#=
1. run
p_opt =
[13.014169886578932
  2.067130614022539
  8.782250563794914
 13.035559079377817
  2.0539846802239246
  8.913880017111973
 12.996417090042813
  2.071839932635213
  8.948344730366804]

2. run
p_opt = 
[12.920528591175692
  2.055521226797314
  7.907112254486979
 12.980761600310702
  2.0513665968206873
  8.426793438967957
 12.961190624776055
  2.0691384801460204
  8.562627674914927]
=#







sol_ref_track = solve(prob,alg,p=p_opt.minimizer, saveat=t_samp)

using Plots
plot(sol_ref_track)

y_out = C*Array(sol_ref_track)
plot(sol_ref_track.t,y_out')
scatter!(sol_ref_track.t,ref_data[1,:])

#=
p_opt = [13.010951664331811
        1.9765036457312297
        8.160716631530311
        12.892699280408907
        2.153289753582373
        8.203973434811285
        12.911826304389534
        2.145243910846501
        8.299341977569908
        12.978783755664848
        1.9883521480910786
        8.285147497006298]
=#


#=
st_pars = hcat(store_param...)'
st_pars =
[13.0652  2.07039  9.07621  13.0652  2.07039  9.07621  13.0652  2.07039  9.07621  13.0652  2.07039  9.07621
 13.0267  2.05703  9.07978  13.0266  2.05731  9.07979  13.0228  2.05889  9.08015  13.0229  2.059    9.08014
 13.0443  2.06228  9.07763  13.0438  2.06257  9.07768  13.0382  2.06439  9.07824  13.0381  2.0644   9.07825
 13.0441  2.06127  9.07721  13.0433  2.06168  9.07729  13.0354  2.06421  9.07809  13.0351  2.06417  9.07813
 13.0512  2.05899  9.07397  13.0486  2.05994  9.07426  13.0294  2.06589  9.07631  13.0282  2.06543  9.07643
 13.052   2.05928  9.07342  13.0491  2.06027  9.07373  13.0299  2.06626  9.07582  13.0287  2.06565  9.07596
 13.0496  2.06233  9.04946  13.0381  2.06555  9.05089  13.0268  2.07227  9.05456  13.0249  2.06319  9.05488
 13.0533  2.06623  8.98528  13.0187  2.07582  8.98971  13.0269  2.0863   8.99762  13.0229  2.05424  8.99845
 13.07    2.04964  8.53293  12.8729  2.10963  8.55863  13.0081  2.16106  8.59739  12.9911  1.96044  8.60171
 13.072   2.03377  8.38715  12.8232  2.11385  8.41969  12.9938  2.18517  8.46903  12.9731  1.92813  8.47435
 13.0822  2.01974  8.30653  12.8034  2.11403  8.34309  12.9891  2.20368  8.39881  12.9652  1.9181   8.40461
 13.0833  2.01216  8.27767  12.7961  2.11338  8.3156   12.9825  2.21329  8.37432  12.9571  1.92145  8.3801
 13.0856  2.00955  8.26339  12.7959  2.11453  8.30202  12.9798  2.21879  8.36271  12.9539  1.92877  8.36828
 13.0845  2.00892  8.26116  12.7957  2.11472  8.29985  12.978   2.21841  8.36105  12.9528  1.93002  8.36647
 13.0836  2.00865  8.25991  12.7961  2.11501  8.29863  12.9768  2.2172   8.36025  12.9524  1.93087  8.3655
 13.0809  2.00793  8.25548  12.7986  2.11634  8.29427  12.9728  2.21201  8.3575   12.9521  1.93419  8.36207
 13.0613  2.00283  8.20998  12.8309  2.13199  8.24977  12.9386  2.1576   8.32946  12.957   1.97053  8.327
 13.0582  2.00152  8.19501  12.8442  2.13754  8.2352   12.9315  2.14147  8.32007  12.9626  1.98382  8.31532
 13.0472  1.99304  8.18437  12.8505  2.13839  8.22469  12.9284  2.13304  8.31282  12.9701  1.99696  8.30603
 13.0435  1.99149  8.18276  12.8508  2.1387   8.2231   12.9269  2.132    8.31174  12.9705  1.99848  8.30456
 13.043   1.99143  8.18207  12.8519  2.13924  8.22245  12.9271  2.13202  8.31127  12.9715  1.999    8.30393
 13.0429  1.99146  8.1819   12.8523  2.13943  8.22229  12.9272  2.13206  8.31116  12.9717  1.99903  8.30378
 13.0429  1.99147  8.18183  12.8524  2.13952  8.22223  12.9272  2.13209  8.31112  12.9718  1.999    8.30371
 13.0418  1.99168  8.18081  12.8547  2.14081  8.22132  12.9269  2.13244  8.31051  12.9727  1.99833  8.30281
 13.0234  1.99039  8.17021  12.875   2.15297  8.21189  12.9176  2.13526  8.30455  12.9762  1.98779  8.29341
 13.0191  1.98301  8.16567  12.8846  2.15302  8.20799  12.9161  2.13919  8.30196  12.9793  1.98722  8.28942
 13.0133  1.97551  8.16212  12.8905  2.15203  8.20498  12.9129  2.14279  8.30003  12.9795  1.9871   8.28631
 13.0119  1.97554  8.1614   12.8916  2.15266  8.20442  12.9121  2.14434  8.29966  12.9791  1.98805  8.2857
 13.0115  1.97597  8.16112  12.8922  2.15303  8.20422  12.9121  2.14489  8.29952  12.979   1.9884   8.28547
 13.0113  1.97622  8.16096  12.8924  2.15318  8.20412  12.912   2.14509  8.29945  12.9789  1.98845  8.28534
 13.011   1.9765   8.16072  12.8927  2.15329  8.20397  12.9118  2.14524  8.29934  12.9788  1.98835  8.28515]
=#

#=
store_loss=[117.51337206249913
            75.89678193445386
            53.064650579443686
            52.770592098620845
            52.24956390007374
            52.20762136422396
            51.52955168691962
            49.92062146920247
            33.66832745230458
            28.875360369664133
            26.45827071210319
            25.351546853746402
            24.97585513383558
            24.858968867378227
            24.779279964942056
            24.532714531367652
            22.74884328599674
            21.706852759068376
            21.01747792027029
            20.957303898367783
            20.903083292539236
            20.895411560442398
            20.8921817702099
            20.870957757167687
            20.739128591078966
            20.398831470340372
            20.15110860801386
            20.09680864975084
            20.083506446306107
            20.079689429887203
            20.074635765135497]
=#