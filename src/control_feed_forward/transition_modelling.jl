
function find_coeffients(N)

    M = zeros(N+1,N+1)
    M[1,:] = ones(N+1)
    for n1=1:N 
        #nr = n1+N;
        idx=N-n1+1
        for n2=1:N+1
            nc=n2+N
            M[n1+1,n2] = factorial(big(nc)) / factorial(big(idx))
            idx+=1
        end
    end

    c = inv(M)*vcat(1,zeros(N))
    return vcat(zeros(N), c)
end

c2 = find_coeffients(2)
c5 = find_coeffients(5)
c10 = find_coeffients(10)

polyn(t,c) = mapreduce(i-> c[i]*t^i,+, 1:length(c))
polyn_der(t,c) = mapreduce(i-> i*c[i]*t^(i-1),+, 1:length(c))

p2(t) = polyn(t,c2)
p5(t) = polyn(t,c5)
p10(t) = polyn(t,c10)

d1p2(t) = polyn_der(t,c2)
d1p5(t) = polyn_der(t,c5)
d1p10(t) = polyn_der(t,c10)

tgrid = 0:0.05:1

polyn(0.1,c2)

using Plots
plot(tgrid, d1p2.(tgrid))
plot!(tgrid, d1p5.(tgrid))
plot!(tgrid, d1p10.(tgrid))

pdata = round.(hcat(p2.(tgrid), p5.(tgrid), p10.(tgrid)), digits=2)
data = hcat(tgrid, pdata)

using DelimitedFiles;
path2folder = "results/data/"
filename = "transition_polynomial.csv"
path2file = path2folder * filename

open(path2file, "w") do io
    writedlm(io, ["t" "x" "y" "z"], ',')
    writedlm(io, data, ',')
end;

filename = "transition_polynomial_derivative.csv"
path2file = path2folder * filename

der_pdata = round.(hcat(d1p2.(tgrid), d1p5.(tgrid), d1p10.(tgrid)), digits=2)
der_data = hcat(tgrid, der_pdata)

open(path2file, "w") do io
    writedlm(io, ["t" "x" "y" "z"], ',')
    writedlm(io, der_data, ',')
end;
