

Nx = 5;
Ny = 9;

# -u[1] + u[2]
# u[i-1] - 2u[i] + u[i]
# u[Nx-1] - u[Nx]

D1 = zeros(Int64,Nx*Ny, Nx*Ny)
D2 = zeros(Int64,Nx*Ny, Nx*Ny)

for j=1:Ny
    di = (j-1)*Nx
    for i=2:Nx-1
        D1[i+di,i-1+di : i+1+di] = [1,-2,1];
    end
    D1[1+di,1+di:2+di] = [-1,1]
    D1[Nx+di,Nx-1+di:Nx+di] = [1,-1]
end

D1

#D2[1+Nx,1:Nx:1+2Nx]
1+(Ny-1)*Nx
for i=1:Nx
    for j=2:Ny-1
        di = (j-1)*Nx
        D2[i+di,i+di-Nx:Nx:(i+di)+Nx] =  [1,-2,1]
        #D2[i+(j-1)*Nx,i-1+(j-1)*Nx : i+1+(j-1)*Nx] = [1,-2,1];
    end
    D2[i,i:Nx:i+Nx] = [-1,1]
    D2[i+(Ny-1)*Nx,i+(Ny-2)*Nx:Nx:i+(Ny-1)*Nx] = [1,-1]
end

a1 = 0.1;
a2 = 1.2;

# D = D1 + D2

D = a1*D1 + a2*D2

sum(D'*D - D*D')

using Plots
spy(D)

using LinearAlgebra
evals_num, evecs_num = eigen(D)

using Plots
scatter(evals_num)

λdata = zeros(Nx,Ny)

for i=1:Nx, j=1:Ny
    λdata[i,j] = -2*a1 + 2a1*cos((i-1)*π / Nx) -2*a2 + 2a2*cos((j-1)*π / Ny)
end

scatter(evals_num)
scatter!(sort(vcat(λdata...)))

ρ=1
evecs_ana = zeros(Nx,Ny,Nx,Ny)
#evecs_ana[:,1] = ones(Nx*Ny) # map(j-> ρ^(j-1), 1:Nx)
#evecs_ana[:,1] = evecs_ana[:,1] / sqrt(sum(evecs_ana[:,1].^2))

for n1 = 1:Nx, n2 = 1:Ny
    for i=1:Nx, j=1:Ny
        evecs_ana[n1,n2,i,j] = cospi.((n1-1)*(2i-1)/(2Nx))*cospi.((n2-1)*(2j-1)/(2Ny))
    end
end


using Plots
surface(evecs_ana[3,1,:,:])

surface(reshape(evecs_num[:,end-2], (Nx,Ny)))


plot(evecs_num[:,end-5])


a,b,c=1,-4,1
# Analytical computation of eigenvalues and eigenvectors
evals_ana = map(k-> b+2*sqrt(a*c) * cos((k-1)*pi / (Nx*Ny)), 1:Nx*Ny)

using Plots
scatter(evals_ana)
scatter!(evals_num[end:-1:1])

scatter(evals_num)