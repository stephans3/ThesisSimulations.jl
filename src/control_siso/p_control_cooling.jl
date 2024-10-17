using Hestia
prop = StaticIsotropic(50, 8000, 400)

L = 0.2
N = 40
rod = HeatRod(L, N)

boundary = Boundary(rod)

h = 10;
Θamb = 300;
ε = 0.2;
em_total = Emission(h, Θamb,ε) # Pure heat radiation
setEmission!(boundary, em_total, :east);

rod_actuation = IOSetup(rod)
scale         = 1.0;  # b=1
input_id      = 1     # Actuator index
pos_actuators = :west # Position of actuators
setIOSetup!(rod_actuation, rod, input_id, scale,  pos_actuators)

function p_controller(y_out)
    K = 4000
    Θdes = 600;
    return max(K*(Θdes-y_out),0)
end

function heat_conduction_total!(dw, w, param, t)
    # u_in = 4467.9*ones(1) # 4e5 * ones(1)    # heat input as vector

    u_in =  p_controller(w[end])*ones(1)
    diffusion!(dw, w, rod, prop, boundary, rod_actuation, u_in)
end


using OrdinaryDiffEq
θinit = 600ones(N) 
tspan = (0.0, 25000)
tsave = tspan[2]/250 # 10.0;
alg = KenCarp5()

prob_total = ODEProblem(heat_conduction_total!,θinit,tspan)
sol_total = solve(prob_total,alg, saveat=tsave)

using Plots

plot(sol_total)
plot(sol_total.t, sol_total[end,:])

plot(sol_total.t, p_controller.(sol_total[end,:]))

################


function pi_controller(e,e_int)
    Kp = 4000
    Ki = 2;
    u = Kp*e + Ki*e_int
    return max(u,0)
end

function heat_conduction_pi!(dw, w, param, t)
    #u_in = 4e5 * ones(1)    # heat input as vector

    dθ = @view dw[1:end-1]
    θ = @view w[1:end-1]

    Θdes = 600;
    e = Θdes-θ[end]
    u_in =  pi_controller(e,w[end])*ones(1)
    diffusion!(dθ, θ, rod, prop, boundary, rod_actuation, u_in)
    dw[end] = e
end



states = vcat(θinit,0)
prob_pi = ODEProblem(heat_conduction_pi!,states,tspan)
sol_pi = solve(prob_pi,alg, saveat=tsave)

dstates = similar(states)
heat_conduction_pi!(dstates, states, 0, 1)

using Plots

plot(sol_pi)
plot(sol_pi.t, sol_pi[end-1,:])


e_p = 600 .- sol_pi[end-1,:]
e_int =  sol_pi[end,:]
u_pi = mapreduce(i->pi_controller(e_p[i],e_int[i]),vcat,1:length(sol_pi.t))

plot(sol_pi.t, u_pi)