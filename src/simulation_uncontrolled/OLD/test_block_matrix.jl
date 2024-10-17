
using LinearAlgebra
N1 = 10;
# A = diagm(collect(1:4))
# B = diagm(collect(5:8))
A = diagm(-4ones(N1))
B = diagm(ones(N1))

for i=2:N1
    A[i-1,i] = 1
    A[i,i-1] = 1
end

Z = zeros(Int64, N1,N1)
T = [A B Z Z Z;
     B A B Z Z;
     Z B A B Z;
     Z Z B A B;
     Z Z Z B A]

evals_T = eigvals(T)
using Plots
scatter(evals_T)


T2 = [-B B Z Z Z;
     B A B Z Z;
     Z B A B Z;
     Z Z B A B;
     Z Z Z B -B]

evals_T2 = eigvals(T2)
scatter(evals_T2)
     
N2 = 5;
λdata = zeros(N1,N2)

a0 = -4
a1 = 1
b0 = 1

for i=1:N1, j=1:N2
    λdata[i,j] = a0 + 2a1*cospi(i / (N1+1)) + 2b0*cospi(j/(N2+1)); 
end


λdata2 = sort(reshape(λdata, N1*N2))

scatter(evals_T)
scatter!(λdata2)















###########
jmax = 7
imax = 13;

mmax = 13;
λdata = zeros(mmax, mmax)

for i=1:mmax, j=1:mmax
    λdata[j,i]λdata = 4-2cos(j*π / (mmax+1))-2cos(i*π / (mmax+1))
end





λdata


mmax = 13;
λdata = zeros(mmax, mmax)

for i=1:mmax, j=1:mmax
    λdata[j,i] = 4-2cos((j-1)*π / mmax)-2cos((i-1)*π / mmax)
end

λdata