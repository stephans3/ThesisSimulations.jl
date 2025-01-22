

Nx = 5;
Ny = 9;
Nz = 11;

# -u[1] + u[2]
# u[i-1] - 2u[i] + u[i]
# u[Nx-1] - u[Nx]

D1 = zeros(Int64,Nx*Ny*Nz, Nx*Ny*Nz)
D2 = zeros(Int64,Nx*Ny*Nz, Nx*Ny*Nz)
D3 = zeros(Int64,Nx*Ny*Nz, Nx*Ny*Nz)

for j=1:Ny, k=1:Nz
    di = (k-1)*Nx*Ny + (j-1)*Nx 
    for i=2:Nx-1
        D1[i+di,i-1+di : i+1+di] = [1,-2,1];
    end
    D1[1+di,1+di:2+di] = [-1,1]
    D1[Nx+di,Nx-1+di:Nx+di] = [1,-1]
end

D1

using Plots
spy(D1)

#D2[1+Nx,1:Nx:1+2Nx]
#1+(Ny-1)*Nx
for i=1:Nx, k=1:Nz
    for j=2:Ny-1
        di = (k-1)*Nx*Ny + (j-1)*Nx 
        #di = (j-1)*Nx
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


using LinearAlgebra
evals_num, evecs_num = eigen(D)

using Plots
scatter(evals_num)


mmax1 = Nx;
mmax2 = Ny;
λdata = zeros(mmax1, mmax2)

for i=1:mmax1, j=1:mmax2
    λdata[i,j] = -(a1*2+a2*2-a2*2cos((j-1)*π / mmax2)-a1*2cos((i-1)*π / mmax1))
end

scatter(evals_num)
scatter!(sort(vcat(λdata...)))