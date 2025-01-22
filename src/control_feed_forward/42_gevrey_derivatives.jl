#=
    Compute first derivatives of transition 
=#

# 1. Derivative = Bump function
dg1(x) = exp(16-1/(x-x^2)^2)

# 2. Derivative
dg2(x) = exp(16-1/(x-x^2)^2)*(4*x-2)/(10*x^6-30*x^5+30*x^4-10*x^3)

# 3. Derivative
dg3(x) = exp(11.15-1/(x-x^2)^2)*(4 - 16*x + 10*x^2 + 32*x^3 - 66*x^4 + 60*x^5 - 20*x^6)/(x^12 - 6*x^11 + 15*x^10 - 20*x^9 + 15*x^8 - 6*x^7 + x^6)

# 4. Derivative
dg4(x) = exp(7.4-1/(x-x^2)^2)*(-8 + 48x - 60x^(2) - 200x^(3) + 756x^(4) - 948x^(5) + 84x^(6) + 1344x^(7) - 2016x^(8) + 1548x^(9) - 660x^(10)  + 120x^(11))/(x^(18) - 9x^(17) + 36x^(16) - 84x^(15) + 126x^(14) - 126x^(13) + 84x^(12) - 36x^(11) + 9x^(10) - x^(9))

x1grid= vcat(0.1,0.2,0.21:0.01:0.79,0.8,0.9)#0.1:0.02:0.9

#Generate dat
dg34_data = hcat(x1grid, round.(dg3.(x1grid),digits=3),round.(dg4.(x1grid),digits=3))


using DelimitedFiles;
path2folder = "results/data/"
filename = "reference_gevrey_deriv_34.csv"
path2file = path2folder * filename

open(path2file, "w") do io
    writedlm(io, dg34_data, ',')
end;
