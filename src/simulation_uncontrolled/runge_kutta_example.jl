

function rk4_classic(t,y,ΔT,a)

    k1 = a*y
    k2 = a*(y+k1*ΔT/2)
    k3 = a*(y+k2*ΔT/2)
    k4 = a*(y+k3*ΔT)

    return y + ΔT*(k1 + 2k2 + 2k3 + k4)/6
end

function rk4_38_rule(t,y,ΔT,a)
    k1 = a*y
    k2 = a*(y+k1*ΔT/3)
    k3 = a*(y+ ΔT*(-k1/3 + k2))
    k4 = a*(y+ ΔT*(k1 -k2 + k3))

    return y + ΔT*(k1 + 3k2 + 3k3 + k4)/8
end


a0 = -1;
y0 = 10;

Ts = 2.6 # 1.6 # 2.79 # 0.1;

y_data = zeros(20,2);
y_data[1,:] .= y0
y_save_1 = y0
y_save_2 = y0

rk4_classic(0,3.1,Ts,a0)
rk4_38_rule(0,3.1,Ts,a0)

for (idx, y) in enumerate(y_data[1:end-1,1])
    y_sol_1 = rk4_classic(0,y_save_1,Ts,a0)
    y_sol_2 = rk4_38_rule(0,y_save_2,Ts,a0)
    y_data[idx+1,1] = y_sol_1
    y_data[idx+1,2] = y_sol_2
    y_save_1 = y_sol_1
    y_save_2 = y_sol_2
end

y_data

using Plots
plot(0:0.1:5, rk4.(0,1,0:0.1:5,-1))
rk4(0,1,2,-1)




function rk44(t,y,ΔT,a)

    k1 = a*y
    k2 = a*(y+k1*ΔT/2)
    k3 = a*(y+k2*ΔT/2)
    k4 = a*(y+k3*ΔT)

    return k1, k2, k3, k4
end