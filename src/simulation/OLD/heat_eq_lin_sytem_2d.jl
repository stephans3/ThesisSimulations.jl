Nx = 10;
Ny = 5;

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

A = D1 + D2

using LinearAlgebra
dA = diag(A)
unique(dA)

subdA = A - diagm(diag(A))
unique(sum(subdA, dims=2))

sum(dA - sum(subdA, dims=2))
dA - sum(subdA, dims=2)

using Plots
scatter(dA)

eigvals(A)