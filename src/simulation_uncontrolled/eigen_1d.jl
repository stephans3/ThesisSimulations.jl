#=
    Computing transposed eigenvectors 

=#

N = 10;
D = zeros(N,N)
a,b,c = 1,-2,1
elem = [a,b,c]
for i=2:N-1
    D[i,i-1 : i+1] = elem;
end
D[1,1:2] = [-1,1];
D[end,end-1:end] = [1,-1];
using LinearAlgebra
e1, v1 = eigen(D)

v1'*v1

A1 = v1 * diagm(e1) * inv(v1)


j = 1:N;
nj = 1:N;
eig_fun(z,n) = cospi((z-1)*(2n-1)/(2N))
v2 = mapreduce(jj -> eig_fun.(jj,nj), hcat, j)
v21 =mapreduce(jj-> v2[:,jj]/norm(v2[:,jj]), hcat, N:-1:1)


inv(v21) - v21'






A2 = v21 * diagm(e1) * inv(v21)

v2' * v2

v3 = mapreduce(jj -> eig_fun.(nj,jj), hcat, j)

eig_fun.(1,j)
eig_fun.(2,j)

eig_fun.(j,1)


v41 = eig_fun.(j',j)
v42 = eig_fun.(j,j')

eig_fun.(2,j)'*eig_fun.(1,j)

function test_uni(jj)
    J = jj[end]

    sol = zeros(J,J)
    for j1 = 1 : J
        for j2 = 1 : J
            sol[j1, j2] = eig_fun.(j2,j)'*eig_fun.(j1,j)
        end
    end
    return sol
end

test_uni(j)

v41*v41'

test_unitary(z,n) = eig_fun(z,n) * eig_fun(z,n)

test_unitary.(j,j')
test_unitary.(j',j)

tu = mapreduce(jj -> test_unitary.(jj,nj), hcat, j)
