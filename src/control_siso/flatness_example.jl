using ControlSystems, Plots

G = tf([1],[1,2,1])

function myinput2(t)
    r = t^3;
    d1r = 3*t^2
    d2r = 6*t
    # d3r = 6

    if t <= Tf
        return r + 2*d1r + d2r
    else
        return 0
    end
end

Tf = 4.0;
tgrid = 0:0.1:Tf
u_data2 = myinput2.(tgrid)
x, _ = lsim(G, u_data2', tgrid, zeros(2));
plot(tgrid, [x[1,:], tgrid.^3],xguide="Time in [s]")

using OrdinaryDiffEq

function myode(dx,x,p,t)
    u = myinput2(t)
    dx[1] = x[2]
    dx[2] = x[1] + 2x[2] + u
end
