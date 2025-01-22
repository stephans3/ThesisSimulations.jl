


nonlinear_ode(x,p,t) = -x - x^4

using OrdinaryDiffEq

x0 = 10.0;
tspan = (0.0, 10.0);
alg = Tsit5();

prob = ODEProblem(nonlinear_ode, x0, tspan)
sol = solve(prob, alg)


using FastGaussQuadrature

x,w = FastGaussQuadrature.gausslegendre(1000)
p1 = Ts/2

# Geht nicht...
function integrate_fun(t, x0)

    myint(τ) = exp(-(t-τ))*(-)

    sol = exp(-1*t)*x0 + int_part

    return sol
end