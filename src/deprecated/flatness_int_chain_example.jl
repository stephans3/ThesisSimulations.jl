
function ref2(t)
    if t < 0
        return 0
    elseif t > Tf
        return 1
    else
        return (1 - cospi(t / Tf)) / 2
    end
end

function der_ref2(t, n)
    if t < 0 || t > Tf
        return 0
    else
        return -cospi(t / Tf + n / 2) / 2 * (pi / Tf)^n
    end
end


function myinput2(t)
    #d2r = der_ref2(t,2)
    d2r = ((π/Tf)^2) * cospi(t/Tf) / 2 # der_ref2(t,2)
    if t <= Tf
        return d2r
    else
        return 0
    end
end

function myode(dx,x,p,t)
    # u =  ((π/Tf)^2) * cospi(t/Tf) / 2

    # u = sinpi(t/Tf) # der_ref2(t, Nstates)
    u = der_ref2(t, Nstates)
    for i=1:Nstates-1
        dx[i] = x[i+1]
    end
    dx[Nstates] = u
    #=
    u = der_ref2(t, Nstates)
    dx[1] = x[2]
    dx[2] = u
    =#
end

Nstates = 3
Tf = 4.0;
# x0 = [(4/π)^3,0.0,-4/π]


x0 = zeros(Nstates) # [0, (π^2)/32]
x0[1] = ref2(0) + 0.2
for i=1:Nstates-1
    x0[i+1] = der_ref2(0, i)
end


tspan = (0, Tf)
using OrdinaryDiffEq
alg = Tsit5()
prob = ODEProblem(myode, x0, tspan)
sol = solve(prob, alg, saveat=0.1)

using Plots
plot(sol,xguide="Time in [s]")


plot(sol.t, der_ref2.(sol.t, Nstates))


using ControlSystems
G = tf([1],[1,0,0])
tgrid = 0:0.1:Tf
u_data2 = myinput2.(tgrid)
x, _ = lsim(G, u_data2', tgrid, zeros(2));

using Plots
plot(tgrid, [x[1,:],ref2.(tgrid)],xguide="Time in [s]")
