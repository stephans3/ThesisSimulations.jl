Nx = 3;
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

B = vcat(1:1:Nx, zeros(Int64,Nx*(Ny-1)))
C = hcat(zeros(Int64,1,Nx*(Ny-1)),(1:1:Nx)')

Nc = Nx*Ny
Om = mapreduce(i-> C*A^i,vcat,0:Nc-1)

rank(Om)

C*B
C*A*B
C*A^2*B
C*A^4*B

B = vcat([1, 0, 0], zeros(Int64,Nx*(Ny-1)))
C = hcat(zeros(Int64,1,Nx*(Ny-1)),[0 0 1])
Om2 = mapreduce(i-> C*A^i,vcat,0:Ny-1)

C*B
C*A*B
C*A^4*B
C*A^4*B

C*A^3


mapreduce(i-> C*A^i,vcat,0:Nx*Ny)


using ControlSystems

sys_ss = ss(A,B,C,[0])
sys_G = tf(sys_ss)

sys_G
