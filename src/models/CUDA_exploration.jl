if pwd() != "/mnt/data/code/ParallelDefault"
    cd("ParallelDefault")
end
using Pkg; Pkg.activate(".")

using Random, Distributions, CUDA, Printf, BenchmarkTools
#Initialization

# -----------------------------------------------------------------------------------

#First Intialze the kernels

#line 8 Calculate Vr, still a double loop inside, tried to flatten out another loop
function vr(Nb,Ny,α,β,τ,Vr,V0,Y,B,Price0,P)

    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)

        Max = -Inf
        for b in 1:Nb
            c = exp(Y[iy]) + B[ib] - Price0[iy,b]*B[b]
            if c > 0 #If consumption positive, calculate value of return
                sumret = 0
                for y in 1:Ny
                    sumret += V0[y,b]*P[iy,y]
                end

                vr = (c^(1-α))/(1-α) + β * sumret
                Max = max(Max, vr)
            end
        end
        Vr[iy,ib] = Max
    end
    return
end


#line 9-14 debt price update
function Decide(Nb,Ny,Vd,Vr,V,decision,decision0,prob,P,Price,rstar)

    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)

        if (Vd[iy] < Vr[iy,ib])
            V[iy,ib] = Vr[iy,ib]
            decision[iy,ib] = 0
        else
            V[iy,ib] = Vd[iy]
            decision[iy,ib] = 1
        end

        for y in 1:Ny
            prob[iy,ib] += P[iy,y] * decision[y,ib]
        end

        Price[iy,ib] = (1-prob[iy,ib]) / (1+rstar)

    end
    return
end


# -----------------------------------------------------------------------------------
#Main starts


#Setting parameters
Ny = 100 #grid number of endowment
Nb = 150 #grid number of bond
maxInd = Ny * Nb #total grid points
rstar = 0.017 #r* used in price calculation
α = 0.5 #α used in utility function
lbd = -1 #lower bound and upper bound for bond initialization
ubd = 0  #lower bound and upper bound for bond initialization
β = 0.953   #β,ϕ,τ used as in part 4 of original paper
ϕ = 0.282   #β,ϕ,τ used as in part 4 of original paper
τ = 0.5 #β,ϕ,τ used as in part 4 of original paper
δ = 0.8 #weighting average of new and old matrixs
ρ = 0.9     #ρ,σ For tauchen method
σ = 0.025   #ρ,σ For tauchen method


#Initializing Bond matrix
minB = lbd
maxB = ubd
step = (maxB-minB) / (Nb-1)
B = CuArray(minB:step:maxB) #Bond

#Intitializing Endowment matrix
σ_z = sqrt((σ^2)/(1-ρ^2))
Step = 10*σ_z/(Ny-1)
Y = CuArray(-5*σ_z:Step:5*σ_z) #Endowment


#Initialize Conditional Probability matrix
#tauchen method for creating conditional probability matrix
function tauchen(ρ, σ, Ny, P)
    #Create equally spaced pts to fill into Z
    σ_z = sqrt((σ^2)/(1-ρ^2))
    Step = 10*σ_z/(Ny-1)
    Z = -5*σ_z:Step:5*σ_z

    #Fill in entries of 1~ny, ny*(ny-1)~ny^2
    for z in 1:Ny
        P[z,1] = cdf(Normal(), (Z[1]-ρ*Z[z] + Step/2)/σ)
        P[z,Ny] = 1 - cdf(Normal(),(Z[Ny] - ρ*Z[z] - Step/2)/σ)
    end

    #Fill in the middle part
    for z in 1:Ny
        for iz in 2:(Ny-1)
            P[z,iz] = cdf(Normal(), (Z[iz]-ρ*Z[z]+Step/2)/σ) - cdf(Normal(), (Z[iz]-ρ*Z[z]-Step/2)/σ)
        end
    end
end
Pcpu = zeros(Ny,Ny)  #Conditional probability matrix
tauchen(ρ, σ, Ny, Pcpu)
P = CuArray(Pcpu)


#Utility function
U(x) = x^(1-α) / (1-α) 


V = CUDA.fill(1/((1-β)*(1-α)),Ny, Nb) #Value
Price = CUDA.fill(1/(1+rstar),Ny, Nb) #Debt price
Vr = CUDA.zeros(Ny, Nb) #Value of good standing
Vd = CUDA.zeros(Ny) #Value of default
decision = CUDA.ones(Ny,Nb) #Decision matrix


## -----------------------------------------------------------------------------------
# This is where the while loop starts!


#Keeping copies of Value, Value of defualt, Price for the previous round
V0 = CUDA.deepcopy(V)
Vd0 = CUDA.deepcopy(Vd)
Price0 = CUDA.deepcopy(Price)
prob = CUDA.zeros(Ny,Nb)
decision = CUDA.ones(Ny,Nb)
decision0 = CUDA.deepcopy(decision)

threadcount = 256 #set up defualt thread numbers per block




## -----------------------------------------------------------------------------------
# Initialise the default value

sumdef = CUDA.zeros(Ny)
tid = CUDA.zeros(Ny)
bid = CUDA.zeros(Ny)
# Intitializing value of default U((1-τ)iy) to each Vd[iy]

function def_init!(sumdef,tid,bid,τ,Y,α)
    iy = threadIdx().x
    stride = blockDim().x
    for i = iy:stride:length(sumdef)
        sumdef[i] = (exp((1-τ)*Y[i])^(1-α))/(1-α)
        tid[i] = threadIdx().x
        bid[i] = blockIdx().x
    end
    return
end
@time @cuda threads=Ny def_init!(sumdef,tid,bid,τ,Y,α)



## -----------------------------------------------------------------------------------
# Use def_add

temp = CUDA.zeros(Ny,Ny)
blockcount = 15

#line 7.2 adding second expected part to calcualte Vd[iy]
function def_add!(matrix, P, β, V0, Vd0, ϕ, Ny)
    y0 = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y1 = (blockIdx().y-1)*blockDim().y + threadIdx().y
    bid = blockIdx().x
    @cuprintln("y0 $y0, y1 $y1, bid $bid")

    if (y0 <= Ny && y1 <= Ny)
        matrix[y0,y1] = β* P[y0,y1]* (ϕ* V0[y1,1] + (1-ϕ)* Vd0[y1])        
    end
    return
end

# blockcount = (ceil(Int,Ny/10), ceil(Int,Ny/10))        
threadcount = (Ny,Ny)
threadcount = 1000

@cuda threads=(10,10) def_add!(temp, P, β, V0, Vd0, ϕ, 10)
CUDA.synchronize()

#Added this part for speed, may not work so well and untidy
temp = sum(temp, dims=2)
Vd = sumdef + temp

#line 8

# blockcount = (ceil(Int,Nb/10),ceil(Int,Ny/10))
@cuda threads=threadcount blocks=blockcount vr(Nb,Ny,α,β,τ,Vr,V0,Y,B,Price0,P)

#line 9-14

# blockcount = (ceil(Int,Nb/10),ceil(Int,Ny/10))
@cuda threads=threadcount blocks=blockcount Decide(Nb,Ny,Vd,Vr,V,decision,decision0,prob,P,Price,rstar)

#line 16
#update Error and value matrix at round end

err = maximum(abs.(V-V0))
PriceErr = maximum(abs.(Price-Price0))
VdErr = maximum(abs.(Vd-Vd0))
Vd = δ * Vd + (1-δ) * Vd0
Price = δ * Price + (1-δ) * Price0
V = δ * V + (1-δ) * V0



##

ny = 100
nb = 150

kernel = @cuda launch=false def_init(CUDA.zeros(ny), 0.5, CuArray(-5*sqrt((0.025^2)/(1-0.025^2)):sqrt((0.025^2)/(1-0.9^2)):5*sqrt((0.025^2)/(1-0.9^2))), 0.5)
config = launch_configuration(kernel.fun)

threads = min(ny*nb, config.threads)
blocks = cld(ny*nb, threads)
# 4992 

##

@time VReturn, VDefault, Decision, Price = main_gpu();
@btime main_gpu();



#-----
#Storing matrices as CSV
#=

using Parsers
using DataFrames
using CSV

dfPrice = DataFrame(Array(Price))
dfVr = DataFrame(Array(VReturn))
dfVd = DataFrame(Array(VDefault))
dfDecision = DataFrame(Array(Decision))

CSV.write("./Price.csv", dfPrice)
CSV.write("./Vr.csv", dfVr)
CSV.write("./Vd.csv", dfVd)
CSV.write("./Decision.csv", dfDecision)