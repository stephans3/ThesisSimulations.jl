
r_in = 2;
r_out = 6;

f_in(n) = r_in*[cos(pi*n/Nin),sin(pi*n/Nin)]
f_out(n,x) = r_out*[cos(pi*n/Nout + x),sin(pi*n/Nout + x)]

Nin = 3;
Nout = 9;
k_set = ((0),(1,2,3),(4,5,6,7,8))

#=
Nin = 2;
Nout = 3;
k_set = ((0),(1,2))
=#


length(k_set)


using LinearAlgebra

function loss(p)
    err = 0
    for n in 0:Nin-1
        for k in k_set[n+1]
           err = err +  norm(f_in(n)-f_out(k,p))^2;
        end
    end
    return err
end

pgrid = -1:0.01:1
loss_data = loss.(pgrid)

using Plots
plot(pgrid, loss_data)

f_in(3)