using LinearAlgebra

function eigvals_he(J, M, K, p)

    i(j,m,k) = j + (m-1)*J + (k-1)*M*J

    Nc = J*M*K
    μ = zeros(Nc)


    for j=1:J, m=1:M,k=1:K
        idx = i(j,m,k);
        μ[idx] = -2p[1]*(1-cospi((j-1)/J)) - 2p[2]*(1-cospi((m-1)/M)) - 2p[3]*(1-cospi((1-k)/K))
    end
    return μ
end

function eigvecs_he(J, M, K;orthonormal = true)

    f(z,n) = cospi((2n-1)*z)

    i(j,m,k) = j + (m-1)*J + (k-1)*M*J
    Nc = J*M*K
    V = zeros(Nc, Nc)

    for j=1:J, m=1:M,k=1:K
        idx = i(j,m,k);
        for n_j=1:J, n_m=1:M, n_k=1:K
            n_idx = i(n_j,n_m,n_k);
           
            ψ = f((j-1)/(2J), n_j)*f((m-1)/(2M),n_m)*f((k-1)/(2K),n_k)
            V[n_idx,idx] = ψ
        end

        if orthonormal == true
            V[:,idx] = V[:,idx] / norm(V[:,idx])
        end
    end
    return V
end





μ = eigvals_he(3, 1, 1, ones(3))
V = eigvecs_he(3, 1, 1)

V'



μ = eigvals_he(3, 1, 1, ones(3))
V = eigvecs_he(3, 1, 1)

V1 = eigvecs_he(3, 1, 1, orthonormal=false)

E1 = [1 0; 0 0; 0 1]
x0 = [1,2,-1]

V'*x0

V1'*E1
V'*E1

V*V'


p=1;

A_trans = p* [-0 0 0;
     0 -1 0;
     0 0 -3]

A_orig = p* [-1 1 0;
     1 -2 1;
     0 1 -1]


V[:,1]
V
A_orig * V[:,1]
μ[1]*V[:,1]


(A_orig*V)'*V
V' * A_orig *V

A_trans_1 = V' * A_orig * V
A_orig_1 = V * A_trans * V'

exp(A_trans_1) ≈ V' * exp(A_orig) * V
exp(A_orig_1) ≈ V * exp(A_trans) * V'

# x'(t) = A x(t) + B u(t)
# z(t) = V x(t)
# z'(t) = V x'(t) = V A V⁻¹ z(t) + V B u(t) = Λ z(t) + B1 u(t)
# x(t) = exp(A t) x(0) + integral(...) = V  exp(Λ t) V⁻¹ x(0) + integral(...)
# z(t) = exp(Λ t) z(0) + integral(...) 


V * exp(A_trans) * V'
exp(A_orig_1)


V' * A_orig * V


V' * A_trans * V


V * exp(A_orig) * V'
V' * exp(A_trans) * V

exp(A_orig)
V * exp(A_trans) * V'
