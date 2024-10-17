A = [-1 1 0;
	1 -2 1;
	0 1 -1]

B = [1, 0, 0]

C = [0 0 1]

T = vcat([  C,
	        C*A,
	        C*(A^2)]...)
Tinv = inv(T)
N = size(A)[1]
Tf = 4;

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


function input_signal2(t)
    c_N = 1 / first(C * A^(N - 1) * B)
    r = ref2(t)
    dr = map(i -> der_ref2(t, i), 1:N)
    r̂ = vcat(r, dr[1:N - 1])
    return c_N * dr[end] - c_N * (C * A^(N) * Tinv * r̂)[1]
end


using ControlSystems, Plots, LinearAlgebra

tgrid = 0:0.1:Tf
plot(tgrid, input_signal2.(tgrid))

u_data = input_signal2.(tgrid)
sys = ss(A,B,C, zeros(1,1))
x, _ = lsim(sys, u_data', tgrid, zeros(3));
plot(tgrid, [x[1,:],ref2.(tgrid)],xguide="Time in [s]")

G = tf(sys)

function myinput(t)
    if t <= Tf
        return (3cos(t)-4sin(t)-cos(t))
    else
        return 0
    end
end

function myinput2(t)
    r = t^3;
    d1r = 3*t^2
    d2r = 6*t
    d3r = 6

    if t <= Tf
        return (3d1r+4d2r+d3r)
    else
        return 0
    end
end

u_data2 = myinput2.(tgrid)
x, _ = lsim(G, u_data2', tgrid, zeros(3));
plot(tgrid, [x[1,:], tgrid.^3],xguide="Time in [s]")

err = tgrid.^3 - x[1,:]
plot(tgrid, err, xguide="Time in [s]")
