λ = 50;  # Thermal conductivity: constant
ρ = 8000;# Mass density: constant
c = 400; # Specific heat capacity: constan
α = λ/(ρ*c) # Diffusivity
L = 0.2
w_tr =0.5;
w_be =0;
dt = 10;
Nx_arr = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

m_max_tr = zeros(length(Nx_arr))
m_max_be = zeros(length(Nx_arr))
for (jn, Nx) in enumerate(Nx_arr)
    dx = L/Nx
    μ_max = -4*α/(dx^2)
    m_max_tr[jn] = (1+w_tr*dt*μ_max) / (1-(1-w_tr)*dt*μ_max) 
    m_max_be[jn] = (1+w_be*dt*μ_max) / (1-(1-w_be)*dt*μ_max) 
end

round.(m_max_tr,digits=3)

round.(m_max_be,digits=3)