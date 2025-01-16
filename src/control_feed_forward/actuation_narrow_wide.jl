
b(x,m) = exp(-(m*(x - 0.05))^4)

num_elements=21;
scale = 30;
xgrid1 = scale*range(0,0.1,num_elements)

xgrid2 = scale*range(0.03,0.07,num_elements)
data1 = map(i-> b(i/scale,30), xgrid1)
data2 = map(i-> b(i/scale,100), xgrid2)

# using Plots
# plot(xgrid1, data1)
# plot!(xgrid2, data2)

# grid2


data1_exp = hcat(xgrid1,round.(data1,digits=5))
data2_exp = hcat(xgrid2,round.(data2,digits=5))

using DelimitedFiles;
path2folder = "results/data/"
filename = "actuation_char_wide.csv"
path2file = path2folder * filename

open(path2file, "w") do io
    writedlm(io, vcat(["t" "b"],data1_exp), ',')
end;

filename = "actuation_char_narrow.csv"
path2file = path2folder * filename
open(path2file, "w") do io
    writedlm(io,  vcat(["t" "b"],data2_exp), ',')
end;