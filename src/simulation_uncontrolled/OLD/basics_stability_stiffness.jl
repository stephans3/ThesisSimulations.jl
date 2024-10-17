# y'(t) = a*y(t)

a = -2
ode(y,t) = a*y

tmax = -2/a
dt = 0.4*tmax;
Tf = 20;
tgrid = 0 : dt : Tf
data = zeros(length(tgrid)+1)
data[1] = 1;
for (i, t) in enumerate(tgrid)
    data[i+1] = data[i] + dt*ode(data[i],t)
end

data