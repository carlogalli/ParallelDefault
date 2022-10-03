if pwd() != "/home/carlo/Documents/ParallelDefault"
    cd("Documents/ParallelDefault")
end
using Pkg; Pkg.activate(".")

using Distributions, oneAPI, Printf, BenchmarkTools, Test, Plots


#= Function definitions =#

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


#line 7.1 Intitializing U((1-τ)iy) to each Vd[iy]
function def_init(sumdef, τ, Y, ϕ, Ny)
    y0 = get_global_id(0)       # thread index
    if (y0 <= Ny)
        sumdef[y0] = (exp((1-τ)*Y[y0])^(1-ϕ))/(1-ϕ)
    end
    return
end

#line 7.2 adding second expected part to calcualte Vd[iy]
function def_add(matrix, P, β, V0, Vd0, θ, Ny)
    y0 = get_global_id(0)
    y1 = get_global_id(1)
    
    if (y0 <= Ny && y1 <= Ny)
        matrix[y0,y1] = β* P[y0,y1]* (θ* V0[y1,1] + (1-θ)* Vd0[y1])
        # matrix[y0,y1] = P[y0,y1]
    end
    return
end

#line 8 Calculate Vr, still a double loop inside, tried to flatten out another loop
function vr(Nb,Ny,ϕ,β,Vr,V0,Y,B,Price0,P)

    # ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    # iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    ib0 = get_global_id(0)
    iy0 = get_global_id(1)
    
    # oneAPI.@println("ib0 $ib0 iy0 $iy0")

    if (ib0 <= Nb && iy0 <= Ny)        

        Max = Float32(-Inf)
        for ib1 in 1:Nb
            c = exp(Y[iy0]) + B[ib0] - Price0[iy0,ib1]*B[ib1]            
            if c > 0 #If consumption positive, calculate value of return
                sumret = 0
                for iy1 in 1:Ny
                    sumret += V0[iy1,ib1]*P[iy0,iy1]
                end

                vr = (c^(1-ϕ))/(1-ϕ) + β * sumret
                Max = max(Max, vr)
            end
        end
        Vr[iy0,ib0] = Max
    end
    return
end


#line 9-14 debt price update
function Decide(Nb,Ny,Vd,Vr,V,decision,prob,P,Price,rstar)

    # ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    # iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    ib0 = get_global_id(0)
    iy0 = get_global_id(1)

    if (ib0 <= Nb && iy0 <= Ny)

        if (Vd[iy0] < Vr[iy0,ib0])
            V[iy0,ib0] = Vr[iy0,ib0]
            decision[iy0,ib0] = Int32(0)
        else
            V[iy0,ib0] = Vd[iy0]
            decision[iy0,ib0] = Int32(1)
        end

        for iy1 in 1:Ny
            prob[iy0,ib0] += P[iy0,iy1] * decision[iy1,ib0]
        end

        Price[iy0,ib0] = (1-prob[iy0,ib0]) / (1+rstar)

    end
    return
end


#= Bellman operator =#
function bo_gpu(V, Vd, Price, decision, Ny, Nb, Y, B, τ, ϕ, P, β, θ, rstar, iter; verbose::Bool=false)
    
    #Keeping copies of Value, Value of defualt, Price for the previous round
    V0 = oneAPI.deepcopy(V)
    Vd0 = oneAPI.deepcopy(Vd)
    Price0 = oneAPI.deepcopy(Price)
    prob = oneAPI.zeros(Float32, Ny,Nb)
    Vr = oneAPI.zeros(Float32, Ny, Nb)
    # decision = oneAPI.ones(Float32, Ny,Nb)
    
    numthreads = 16
    threadcount = (numthreads, numthreads) #set up default thread numbers per block
    blockcount_yy = (cld(Ny, numthreads), cld(Ny, numthreads))
    blockcount_by = (cld(Nb, numthreads), cld(Ny, numthreads))

    #Initialise the default value
    sumdef = oneAPI.zeros(Float32, Ny)
    @oneapi items=Ny def_init(sumdef,τ,Y,ϕ,Ny)

    #Compute the 2nd part of the default value
    temp = oneAPI.zeros(Float32, Ny,Ny)
    
    @oneapi items=threadcount groups=blockcount_yy def_add(temp, P, β, V0, Vd0, θ, Ny)

    #Added this part for speed, may not work so well and untidy
    temp = sum(temp, dims=2)
    Vd = sumdef + temp

    #Compute the repayment value
    @oneapi items=threadcount groups=blockcount_by vr(Nb,Ny,ϕ,β,Vr,V0,Y,B,Price0,P)

    #Compute price and decision rules
    @oneapi items=threadcount groups=blockcount_by Decide(Nb,Ny,Vd,Vr,V,decision,prob,P,Price,rstar)

    
    #Update error and value matrix at round end
    err = maximum(abs.(V-V0))
    # PriceErr = maximum(abs.(Price-Price0))
    # VdErr = maximum(abs.(Vd-Vd0))
    # Vd = δ * Vd + (1-δ) * Vd0
    # Price = δ * Price + (1-δ) * Price0
    # V = δ * V + (1-δ) * V0

    if verbose
        println(@sprintf("Errors of round %.0f: Value error: %.2e, price error: %.2e, Vd error: %.2e", iter, err, PriceErr, VdErr))
    end

    # return err
    return nothing
end


#= Main starts =#
function main_gpu(; verbose::Bool=false, maxIter::Int=300)

    #Setting parameters
    Ny = Int32(21)     #grid number of endowment
    Nb = Int32(100)     #grid number of bond
    # maxInd = Ny * Nb    #total grid points
    rstar = Float32(0.017) #r* used in price calculation
    lbd = Float32(-1)   #lower bound and upper bound for bond initialization
    ubd = Float32(0)    #lower bound and upper bound for bond initialization
    β = Float32(0.953)  #β,θ,τ used as in part 4 of original paper
    θ = Float32(0.282)  #β,θ,τ used as in part 4 of original paper
    ϕ = Float32(0.5)    #ϕ used in utility function
    δ = Float32(0.8)   #updating weight of new matrix
    ρ = Float32(0.9)    #ρ,σ For tauchen method
    σ = Float32(0.025)  #ρ,σ For tauchen method
    τ = Float32(0.5)    #β,θ,τ used as in part 4 of original paper


    #Initializing Bond matrix
    minB = lbd
    maxB = ubd
    step = (maxB-minB) / (Nb-1)
    B = oneArray{Float32}(minB:step:maxB) #Bond grid

    #Intitializing Endowment matrix
    σ_z = sqrt((σ^2)/(1-ρ^2))
    Step = 10*σ_z/(Ny-1)
    Y = oneArray{Float32}(-5*σ_z:Step:5*σ_z) #Endowment

    #Conditional Probability matrix
    Pcpu = zeros(Ny,Ny)
    tauchen(ρ, σ, Ny, Pcpu)
    P = oneArray{Float32}(Pcpu)

    #Utility function
    U(x) = x^(1-ϕ) / (1-ϕ) 

    #Initialise arrays
    V = oneArray{Float32}(fill(1/((1-β)*(1-ϕ)), Ny, Nb))    #Value
    Price = oneArray{Float32}(fill(1/(1+rstar), Ny, Nb))     #Debt price
    Vr = oneAPI.zeros(Float32, Ny, Nb)                      #Value of good standing
    Vd = oneAPI.zeros(Float32, Ny)                          #Value of default
    decision = oneAPI.ones(Float32, Ny,Nb)                  #Decision matrix

    err = 2000 #error
    tol = 1e-6 #error toleration
    iter = 0


    #Based on Paper Part4, Sovereign meets C++

    while (err > tol) & (iter < maxIter)
        #Keeping copies of Value, Value of defualt, Price for the previous round
        V0 = oneAPI.deepcopy(V)
        Vd0 = oneAPI.deepcopy(Vd)
        Price0 = oneAPI.deepcopy(Price)
        prob = oneAPI.zeros(Float32, Ny,Nb)
        # decision = oneAPI.ones(Float32, Ny,Nb)
        
        numthreads = 16
        threadcount = (numthreads, numthreads) #set up default thread numbers per block

        #Initialise the default value
        sumdef = oneAPI.zeros(Float32, Ny)
        @oneapi items=Ny def_init(sumdef,τ,Y,ϕ,Ny)

        #Compute the 2nd part of the default value
        temp = oneAPI.zeros(Float32, Ny,Ny)
        blockcount = (cld(Ny, numthreads), cld(Ny, numthreads))
        @oneapi items=threadcount groups=blockcount def_add(temp, P, β, V0, Vd0, θ, Ny)

        #Added this part for speed, may not work so well and untidy
        temp = sum(temp,dims=2)
        Vd = sumdef + temp

        #Compute the repayment value
        blockcount = (cld(Nb, numthreads), cld(Ny, numthreads))
        @oneapi items=threadcount groups=blockcount vr(Nb,Ny,ϕ,β,Vr,V0,Y,B,Price0,P)

        #Compute price and decision rules
        blockcount = (cld(Nb, numthreads), cld(Ny, numthreads))
        @oneapi items=threadcount groups=blockcount Decide(Nb,Ny,Vd,Vr,V,decision,prob,P,Price,rstar)

        
        #Update error and value matrix at round end
        err = maximum(abs.(V-V0))
        PriceErr = maximum(abs.(Price-Price0))
        VdErr = maximum(abs.(Vd-Vd0))
        Vd = δ * Vd + (1-δ) * Vd0
        Price = δ * Price + (1-δ) * Price0
        V = δ * V + (1-δ) * V0

        iter += 1
        if verbose
            println(@sprintf("Errors of round %.0f: Value error: %.2e, price error: %.2e, Vd error: %.2e", iter, err, PriceErr, VdErr))
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

@time VReturn_gpu, VDefault_gpu, Decision_gpu, Price_gpu = main_gpu(verbose=false, maxIter=250);

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