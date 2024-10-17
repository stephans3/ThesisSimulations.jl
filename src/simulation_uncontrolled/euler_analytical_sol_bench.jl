#=
d/dt x(t) = a x(t), x(0)=1

x(t) = exp(a t)
=#
 
a = -1;
x0 = 1;
x(t) = exp(a*t) * x0

euler_forward(n,x0,ΔT) = (1+a*ΔT)^n * x0
euler_backward(n,x0,ΔT) = (1-a*ΔT)^(-n) * x0
trapezoidal_rule(n,x0,ΔT) =  ((1+a*ΔT/2)/(1-a*ΔT/2))^(n) * x0



tgrid = 0 : 0.1 : 5;
ngrid = 0 : 1 : 10;
ΔT = 2.

data_ef = euler_forward.(ngrid,x0,ΔT)
data_eb = euler_backward.(ngrid,x0,ΔT)
data_tr = trapezoidal_rule.(ngrid,x0,ΔT)

using Plots
plot(tgrid, x.(tgrid))
# scatter!(ΔT*ngrid, data_ef)
# scatter!(ΔT*ngrid, data_eb)
scatter!(ΔT*ngrid, data_tr)