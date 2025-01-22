

Nx_arr = [10,20,40,60,80,100,200,400,600,800,1000];

err_vals_store = zeros(length(Nx_arr))
err_vecs_store = zeros(length(Nx_arr))

for (n, Nx) in enumerate(Nx_arr)
    D = zeros(Int64,Nx, Nx)
    a,b,c = 1,-2,1
    elem = [a,b,c]
    for i=2:Nx-1
        D[i,i-1 : i+1] = elem;
    end
    D[1,2] = c;
    D[end,end-1] = a;

    α = β = -sqrt(a*c)
    D[1,1] = b-α;
    D[end,end] = b-α;


    using LinearAlgebra
    # Numerical computation of eigenvalues and eigenvectors
    evals_num, evecs_num = eigen(D)

    # Analytical computation of eigenvalues and eigenvectors
    evals_ana = map(k-> b+2*sqrt(a*c) * cos((k-1)*pi / Nx), 1:Nx)

    ρ = sqrt(a/c)
    evecs_ana = zeros(Nx,Nx)
    evecs_ana[:,1] = map(j-> ρ^(j-1), 1:Nx)
    evecs_ana[:,1] = evecs_ana[:,1] / sqrt(sum(evecs_ana[:,1].^2))
    for k = 2 :Nx
        evecs_ana[:,k] = map(j-> ρ^(j-1) * cos((k-1)*(2j-1)*pi/(2Nx)), 1:Nx)
        evecs_ana[:,k] = evecs_ana[:,k] / sqrt(sum(evecs_ana[:,k].^2))
    end

    # Error between analytical and numerical eigenvalues and eigenvectors
    err_evals = evals_ana - evals_num[end:-1:1]

    evecs_ana_pos = sqrt.(evecs_ana.*evecs_ana)
    evecs_num_pos = sqrt.(evecs_num[:,end:-1:1].*evecs_num[:,end:-1:1])
    err_evecs = sqrt.( sum((evecs_ana_pos - evecs_num_pos).^2, dims=1))

    err_vals_store[n] = sum(err_evals) / Nx
    err_vecs_store[n] = sum(err_evecs) / Nx
end

plot(err_vals_store)
plot(err_vecs_store)

