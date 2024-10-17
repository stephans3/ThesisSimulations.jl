
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

E₁₀ = zeros(Int64,Nx,2)
E₁₀[1,1] = 1;
E₁₀[end,2] = 1;


J = 3# Nx;
M = 4# Ny;
K = 5
Nc = J*M*K

E1 = zeros(Int64,Nc,2M*K);
E2 = zeros(Int64,Nc,2J*K);
E3 = zeros(Int64,Nc,2J*M);

for n=1:M*K

    i11 = J*(n-1) + 1
    i12 = 2n-1
    
    i21 = J*n    
    i22 = 2n

    E1[i11, i12] = 1;
    E1[i21, i22] = 1;
end

E1*E1'

using LinearAlgebra

for k=1:K
    # (1,1) - (J,J)
    # ([M-1]*J+1,J+1) -- (M*J,2J)

    # M*J*[k-1]+1 -> M*J*[k-1]+1 + (J-1) = M*J*k -M*J +1 +J -1 =  M*J*k -M*J +J = J*(M*[k-1]+1) 
    # 2*J*[k-1]+1 -> 2*J*[k-1]+1 + (J-1) = 2*J*k - 2*J + 1 + J -1 = J*(2*k-1)

    di = J-1
    i11 = M*J*(k-1)+1
    i12 = 2*J*(k-1)+1
    i21 = J*(M*k-1)+1 #J*(M-1)*k+1
    i22 = J*(2k-1)+1 
    E2[i11:i11+di,i12:i12+di] = diagm(ones(Int64,J))
    E2[i21:i21+di,i22:i22+di] = diagm(ones(Int64,J))
end

M_erg = E2*E2'

Bm =rand(Nc,Nc)*M_erg

#E2*diagm(vcat(ones(15),zeros(15)))*E2'

function bc_blocks(J,M,K)
    Nc = J*M*K

    E1 = zeros(N,2M*K);
    E2 = zeros(N,2J*K);
    E3 = zeros(N,2J*M);



    return E1, E2, E3
end

using LinearAlgebra
diagm(E₁₀,E₁₀)

bm = BlockDiagonal(E₁₀,E₁₀,E₁₀)