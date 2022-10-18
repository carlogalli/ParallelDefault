if pwd() != "/home/carlo/Documents/ParallelDefault"
    cd("Documents/ParallelDefault")
end
using Pkg; Pkg.activate(".")

using Distributions, oneAPI, Printf, BenchmarkTools, Test, Plots

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

struct ModelGPU
    Ny::Int32
    Nb::Int32
    rstar::Float32
    lbd::Float32
    ubd::Float32
    β::Float32
    θ::Float32
    ϕ::Float32
    δ::Float32
    ρ::Float32
    σ::Float32
    τ::Float32
end

function ModelGPU(; Ny=21, Nb=100, rstar=0.017, lbd=-1, ubd=0, β=0.953, θ=0.282, ϕ=0.5, δ=0.8, ρ=0.9, σ=0.025, τ=0.5)

    return ModelGPU(Ny, Nb, rstar, lbd, ubd, β, θ, ϕ, δ, ρ, σ, τ)
end

#Utility function
U(m::ModelGPU, x) = x^(1-m.ϕ) / (1-m.ϕ) 

#line 7.1 Intitializing U((1-τ)iy) to each Vd[iy]
function def_init(sumdef, Y, m)
    y0 = get_global_id(0)       # thread index
    if (y0 <= m.Ny)
        sumdef[y0] = (exp((1-m.τ)*Y[y0])^(1-m.ϕ))/(1-m.ϕ)
    end
    return
end

#line 7.2 adding second expected part to calcualte Vd[iy]
function def_add(matrix, P, V0, Vd0, m)
    y0 = get_global_id(0)
    y1 = get_global_id(1)
    
    if (y0 <= m.Ny && y1 <= m.Ny)
        matrix[y0,y1] = m.β* P[y0,y1]* (m.θ* V0[y1,1] + (1-m.θ)* Vd0[y1])
    end
    return
end

#line 8 Calculate Vr, still a double loop inside, tried to flatten out another loop
function vr(m,Vr,V0,Y,B,Price0,P)

    # ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    # iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    ib0 = get_global_id(0)
    iy0 = get_global_id(1)
    
    # oneAPI.@println("ib0 $ib0 iy0 $iy0")

    if (ib0 <= m.Nb && iy0 <= m.Ny)        

        Max = Float32(-Inf)
        for ib1 in 1:m.Nb
            c = exp(Y[iy0]) + B[ib0] - Price0[iy0,ib1]*B[ib1]            
            if c > 0 #If consumption positive, calculate value of return
                sumret = 0
                for iy1 in 1:m.Ny
                    sumret += P[iy0,iy1]*V0[iy1,ib1]
                end

                vr = (c^(1-m.ϕ))/(1-m.ϕ) + m.β * sumret
                Max = max(Max, vr)
            end
        end
        Vr[iy0,ib0] = Max
    end
    return
end


#line 9-14 debt price update
function decide(Vd,Vr,V,decision,prob,P,Price,m)

    # ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    # iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    ib0 = get_global_id(0)
    iy0 = get_global_id(1)

    if (ib0 <= m.Nb && iy0 <= m.Ny)

        if (Vd[iy0] < Vr[iy0,ib0])
            V[iy0,ib0] = Vr[iy0,ib0]
            decision[iy0,ib0] = Int32(0)
        else
            V[iy0,ib0] = Vd[iy0]
            decision[iy0,ib0] = Int32(1)
        end

        for iy1 in 1:m.Ny
            prob[iy0,ib0] += P[iy0,iy1] * decision[iy1,ib0]
        end

        Price[iy0,ib0] = (1-prob[iy0,ib0]) / (1+m.rstar)
    end
    return
end

function initModelGPU(m::ModelGPU)
    
    #Initializing Bond matrix
    minB = m.lbd; maxB = m.ubd; step = (maxB-minB) / (m.Nb-1)
    B = oneArray{Float32}(minB:step:maxB) #Bond grid

    #Intitializing Endowment matrix
    σ_z = sqrt((m.σ^2)/(1-m.ρ^2)); Step = 10*σ_z/(m.Ny-1)
    Y = oneArray{Float32}(-5*σ_z:Step:5*σ_z) #Endowment

    #Conditional Probability matrix
    Pcpu = zeros(m.Ny,m.Ny)
    tauchen(m.ρ, m.σ, m.Ny, Pcpu)
    P = oneArray{Float32}(Pcpu)

    #Initialise arrays
    V = oneArray{Float32}(fill(1/((1-m.β)*(1-m.ϕ)), m.Ny, m.Nb))    #Value
    Price = oneArray{Float32}(fill(1/(1+m.rstar), m.Ny, m.Nb))     #Debt price
    Vr = oneAPI.zeros(Float32, m.Ny, m.Nb)                      #Value of good standing
    Vd = oneAPI.zeros(Float32, m.Ny)                          #Value of default
    decision = oneAPI.ones(Float32, m.Ny, m.Nb)                  #Decision matrix
    sumdef = oneAPI.zeros(Float32, m.Ny)
    tempVd = oneAPI.zeros(Float32, m.Ny, m.Ny)
    prob = oneAPI.zeros(Float32, m.Ny,m.Nb)

    return V, Vr, Vd, Price, decision, sumdef, tempVd, prob, P, Y, B
end

#= Bellman operator =#
function bo_gpu(m, V, Vr, Vd, Price, decision, sumdef, tempVd, prob, P, Y, B, openapi_params)
    
    threadcount, blockcount_yy, blockcount_by = openapi_params

    # #Initialise the default value
    # @oneapi items=m.Ny def_init(sumdef, Y, m)
    # #Compute the 2nd part of the default value        
    # @oneapi items=threadcount groups=blockcount_yy def_add(tempVd, P, V, Vd, m)
    # #Added this part for speed, may not work so well and untidy
    # tempVd = reduce(+, tempVd, dims=2)
    # Vd .= sumdef + tempVd

    pv = @. (exp((1-mGPU.τ)*Y_gpu)^(1-mGPU.ϕ))/(1-mGPU.ϕ)
    fv =  @. (m.θ * V[:,1] + (1-m.θ)* Vd[:])
    Vd .= pv + m.β * P * fv
    # @show Vd, pv, fv

    #Compute the repayment value
    @oneapi items=threadcount groups=blockcount_by vr(m,Vr,V,Y,B,Price,P)

    #Compute price and decision rules
    @oneapi items=threadcount groups=blockcount_by decide(Vd,Vr,V,decision,prob,P,Price,m)

    return nothing
end

function main_gpu(m::ModelGPU; verbose::Bool=false, maxIter::Int=300)

    V, Vr, Vd, Price, decision, sumdef, tempVd, prob, P, Y, B = initModelGPU(m)

    err = 2000 #error
    tol = 1e-6 #error toleration
    iter = 0

    numthreads = 16
    threadcount = (numthreads, numthreads) #set up default thread numbers per block
    blockcount_yy = (cld(m.Ny, numthreads), cld(m.Ny, numthreads))
    blockcount_by = (cld(m.Nb, numthreads), cld(m.Ny, numthreads))
    openapi_params = threadcount, blockcount_yy, blockcount_by
    
    while (err > tol) & (iter < maxIter)
        #Keeping copies of Value, Value of defualt, Price for the previous round
        V0 = oneAPI.deepcopy(V)

        bo_gpu(m, V, Vr, Vd, Price, decision, sumdef, tempVd, prob, P, Y, B, openapi_params)

        #Update error and value matrix at round end
        err = maximum(abs.(V-V0))
        iter += 1

        if verbose
            # println(@sprintf("Errors of round %.0f: Value error: %.2e, price error: %.2e, Vd error: %.2e", iter, err, PriceErr, VdErr))
            println(@sprintf("Errors of round %.0f: Value error: %.2e", iter, err))
        end
    end

    #Print final results
    println("Total Round ",iter, " error ", err)

    # Vd = Vd[:,:]

    if verbose
        println("Vr: ====================")
        display(Vr)
        println("Vd: ==================")
        display(Vd)
        println("Decision: ==================")
        display(decision)
        println("Price: ==================")
        display(Price)
    end

    return Vr, Vd, decision, Price

end


##

mGPU = ModelGPU(Ny=21, Nb=100, rstar=0.017, lbd=-1, ubd=0, β=0.953, θ=0.282, ϕ=0.5, δ=0.8, ρ=0.9, σ=0.025, τ=0.5)

V_gpu, Vr_gpu, Vd_gpu, Price_gpu, decision_gpu, sumdef_gpu, tempVd_gpu, prob_gpu, P_gpu, Y_gpu, B_gpu = initModelGPU(mGPU)

numthreads = 16; threadcount = (numthreads, numthreads); blockcount_yy = (cld(mGPU.Ny, numthreads), cld(mGPU.Ny, numthreads)); blockcount_by = (cld(mGPU.Nb, numthreads), cld(mGPU.Ny, numthreads)); openapi_params = threadcount, blockcount_yy, blockcount_by

bo_gpu(mGPU, V_gpu, Vr_gpu, Vd_gpu, Price_gpu, decision_gpu, sumdef_gpu, tempVd_gpu, prob_gpu, P_gpu, Y_gpu, B_gpu, openapi_params)


##

@btime bo_gpu(mGPU, V_gpu, Vr_gpu, Vd_gpu, Price_gpu, decision_gpu, sumdef_gpu, tempVd_gpu, prob_gpu, P_gpu, Y_gpu, B_gpu, openapi_params)
# 314 μs
# 676 μs with the broadcasting stuff


##
#= see if and where the values diverge =#

mGPU = ModelGPU(Ny=5, Nb=7, rstar=0.017, lbd=-1, ubd=0, β=0.953, θ=0.282, ϕ=0.5, δ=0.8, ρ=0.9, σ=0.025, τ=0.5)

V_gpu, Vr_gpu, Vd_gpu, Price_gpu, decision_gpu, sumdef_gpu, tempVd_gpu, prob_gpu, P_gpu, Y_gpu, B_gpu = initModelGPU(mGPU)

numthreads = 16; threadcount = (numthreads, numthreads); blockcount_yy = (cld(mGPU.Ny, numthreads), cld(mGPU.Ny, numthreads)); blockcount_by = (cld(mGPU.Nb, numthreads), cld(mGPU.Ny, numthreads)); openapi_params = threadcount, blockcount_yy, blockcount_by

# Price_gpu
# B_gpu

bo_gpu(mGPU, V_gpu, Vr_gpu, Vd_gpu, Price_gpu, decision_gpu, sumdef_gpu, tempVd_gpu, prob_gpu, P_gpu, Y_gpu, B_gpu, openapi_params)

Vr_gpu


## debug vr

function loop_gpu(m, iy0, ib0)
    
    Max = Float32(-Inf)
    for ib1 in 1:m.Nb
        @show c = exp(Y_gpu[iy0]) + B_gpu[ib0] - Price_gpu[iy0,ib1]*B[ib1]            
        if c > 0 #If consumption positive, calculate value of return
            sumret = 0
            for iy1 in 1:m.Ny
                sumret += P_gpu[iy0,iy1]*V_gpu[iy1,ib1]
            end

            vr = (c^(1-m.ϕ))/(1-m.ϕ) + m.β * sumret
            Max = max(Max, vr)
        end
    end
    return Max
end

iy0 = mGPU.Ny
ib0 = mGPU.Nb
loop_gpu(mGPU, iy0, ib0)




##

@time VReturn_gpu, VDefault_gpu, Decision_gpu, Price_gpu = main_gpu(verbose=false, maxIter=275);

p1 = plot([VReturn_gpu[:,50], VDefault_gpu])
p2 = plot([VReturn[:,50], VDefault])
p3 = plot(Decision_gpu[:,50])
p4 = plot(Decision[:,50])
p5 = plot(Price_gpu[:,50])
p6 = plot(Price[:,50])

plot(p1,p2,p3,p4,p5,p6, layout=(3,2))

##

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




## old bellman_operator GPU


#Keeping copies of Value, Value of defualt, Price for the previous round
V0 = oneAPI.deepcopy(V)

Vd0 = oneAPI.deepcopy(Vd)
Price0 = oneAPI.deepcopy(Price)


#Initialise the default value
@oneapi items=Ny def_init(sumdef, m)

#Compute the 2nd part of the default value        
@oneapi items=threadcount groups=blockcount_yy def_add(tempVd, P, β, V0, Vd0, θ, Ny)

#Added this part for speed, may not work so well and untidy
tempVd = reduce(+, tempVd, dims=2)
Vd = sumdef .+ tempVd

#Compute the repayment value
@oneapi items=threadcount groups=blockcount_by vr(Nb,Ny,ϕ,β,Vr,V0,Y,B,Price0,P)

#Compute price and decision rules
blockcount = (cld(Nb, numthreads), cld(Ny, numthreads))
@oneapi items=threadcount groups=blockcount_by Decide(Nb,Ny,Vd,Vr,V,decision,prob,P,Price,rstar)


#Update error and value matrix at round end
err = maximum(abs.(V-V0))
# PriceErr = maximum(abs.(Price-Price0))
# VdErr = maximum(abs.(Vd-Vd0))
# Vd = δ * Vd + (1-δ) * Vd0
# Price = δ * Price + (1-δ) * Price0
# V = δ * V + (1-δ) * V0
iter += 1

if verbose
    println(@sprintf("Errors of round %.0f: Value error: %.2e, price error: %.2e, Vd error: %.2e", iter, err, PriceErr, VdErr))
end

=#