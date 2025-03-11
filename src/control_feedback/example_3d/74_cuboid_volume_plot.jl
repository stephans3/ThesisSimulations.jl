L = 0.2; # Length 
W = 0.2; # Width
H = 0.05;# Height
N₁ = 10;
N₂ = 10;
N₃ = 5;
Nc = N₁*N₂*N₃ 
Δx₁ = L/N₁
Δx₂ = W/N₂
Δx₃ = H/N₃

x1grid = 100*( Δx₁/2 : Δx₁ : L)
x2grid = 100*(Δx₂/2 : Δx₂ : W)
x3grid = 100*(Δx₃/2 : Δx₃ : H)

using DelimitedFiles;
path2folder = "results/data/"
filename = "example_cuboid_mpc.csv"
path2file = path2folder * filename

data = readdlm(path2file, ',')

data_3d = reshape(data[:,end],N₁,N₂,N₃)

using GLMakie
path2folder_pdf = "results/figures/controlled/cuboid_example/"
begin
    filename_pdf = path2folder_pdf * "/cuboid_mpc_3d.png"
    fig = Figure(size=(1200,900),fontsize=26)
    # ax1 = Axis3(fig[1, 1], aspect = (1, 1, 1),xlabel = "Length in [cm]", ylabel = "Width in [cm]", zlabel = "Height in [cm]")
    ax1 = Axis3(fig[1, 1], aspect = (1, 1, 1),xlabel = L"Length $x_{1}$ in [cm]", ylabel = L"Width $x_{2}$ in [cm]", zlabel = L"Height $x_{3}$ in [cm]", xlabelsize = 35, ylabelsize = 35, zlabelsize = 35)
    # ax1 = Axis3(fig[1, 1], perspectiveness = 0.5, azimuth = 7.19, elevation = 0.57, aspect = (1, 1, 1))
    vol = volume!(ax1, x1grid, x2grid, x3grid, data_3d, colormap = :plasma)
    Colorbar(fig[1, 2], vol ,  ticks = collect(495:1:503))
    fig    
    save(filename_pdf, fig, pt_per_unit = 1)       
end
    

