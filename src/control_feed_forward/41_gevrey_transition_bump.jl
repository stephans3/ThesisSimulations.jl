#=
    Compute transition and bump function
=#

using FastGaussQuadrature
t_gq, weights_gq = FastGaussQuadrature.gausslegendre(10000)

function bump_fun(t; T=1, w=2)

    if (t<=0) || (t>=T)
        return 0;
    else
        return exp(-1 / (t/T - (t/T)^2)^w)
    end
end

function transition_fun(t; T=1, w=2)
    if t<= 0
        return 0;
    elseif t >= T
        return 1;
    else
        q = t/2;
        p = T/2;
        Ω_num = q *FastGaussQuadrature.dot( weights_gq ,bump_fun.(q*t_gq .+ q,T=T, w=w))
        Ω_den = p *FastGaussQuadrature.dot( weights_gq ,bump_fun.(p*t_gq .+ p,T=T, w=w))
        return Ω_num / Ω_den 
    end

end

Tf = 1;
J = 40;
tgrid = 0.0 : Tf/J : Tf;
p = Tf/2;
# w_vec = [11,15,20,25,30]
w_vec = [11,20,30]
# Transition
ψ = zeros(length(tgrid), length(w_vec))
Ω = zeros(length(tgrid), length(w_vec))



for (idx, w) in enumerate(w_vec)
    # Original function
    ψ[:,idx] = transition_fun.(tgrid,T=Tf,w=0.1*w) 

    # First derivative
    Ω_den = p *FastGaussQuadrature.dot( weights_gq ,bump_fun.(p*t_gq .+ p,T=Tf, w=0.1*w))
    Ω_num = bump_fun.(tgrid,T=Tf, w=0.1*w)
    Ω[:,idx] = Ω_num / Ω_den
end

ψdata = hcat(tgrid, round.(ψ,digits=3))
Ωdata =hcat(tgrid, round.(Ω,digits=3))

plot(tgrid,Ωdata)

idx=4
maximum(Ωdata[:,idx])
1/maximum(Ωdata[:,idx])

using DelimitedFiles;
path2folder = "results/data/"
filename = "gevrey_transition.csv"
path2file = path2folder * filename

open(path2file, "w") do io
    writedlm(io, ["t" "x" "y" "z"], ',')
    writedlm(io, ψdata, ',')
end;

filename = "gevrey_bump.csv"
path2file = path2folder * filename

open(path2file, "w") do io
    writedlm(io, ["t" "x" "y" "z"], ',')
    writedlm(io, Ωdata, ',')
end;