Nx = 5;
Ny = 3;

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

a1 = 1 # 0.1;
a2 = 2 # 1.2;

# D = D1 + D2

A = a1*D1 + a2*D2

using LinearAlgebra
B = vcat(diagm(1:1:Nx), zeros(Int64,Nx*(Ny-1),Nx))
C = hcat(zeros(Int64,Nx,Nx*(Ny-1)),diagm(1:1:Nx))

C*A^2*B

A^2
C*A
A^2

Nc = Nx*Ny
Om = mapreduce(i-> C*A^i,vcat,0:Nc-1)
