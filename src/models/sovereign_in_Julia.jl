# if pwd() != "/mnt/data/code/ParallelDefault"
#     cd("ParallelDefault")
# end
if pwd() != "/home/carlo/Documents/ParallelDefault"
    cd("Documents/ParallelDefault")
end
using Pkg; Pkg.activate(".")

using Random, Distributions, Printf, BenchmarkTools, Plots


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

function bo_cpu(V, Vr, Vd, Price, decision,
                Ny, Nb, Y, B, 
                τ, θ, β, rstar)

    V0 = deepcopy(V)
    Vd0 = deepcopy(Vd)
    Price0 = deepcopy(Price)
    prob = zeros(Ny, Nb)

    for ib in 1:Nb
        for iy = 1:Ny


            #compute default and repayment
            #7
            
            sumdef = U(exp((1-τ)*Y[iy]))
            for y in 1:Ny
                sumdef += (β* P[iy,y]* (θ* V0[y,1] + (1-θ)* Vd0[y]))
            end
            Vd[iy] = sumdef

            #8

            Max = -Inf
            for b in 1:Nb
                c = exp(Y[iy]) + B[ib] - Price0[iy,b]*B[b]
                if c > 0
                    sumret = 0
                    for y in 1:Ny
                        sumret += P[iy,y]*V0[y,b]
                    end
                    vr = U(c) + β * sumret
                    Max = max(Max, vr)
                end
            end
            Vr[iy,ib] = Max


            #Choose repay or default
            if (Vd[iy] < Vr[iy,ib])
                V[iy,ib] = Vr[iy,ib]
                decision[iy,ib] = 0
            else
                V[iy,ib] = Vd[iy]
                decision[iy,ib] = 1
            end

            #calculate debt price
            for y in 1:Ny
                prob[iy,ib] += P[iy,y] * decision[y,ib]
            end
            Price[iy,ib] = (1-prob[iy,ib]) / (1+rstar)

        end
    end
end

function main_cpu(; verbose::Bool=false, maxIter::Int=300)

    Ny = 21
    Nb = 100
    maxInd = Ny * Nb
    rstar = 0.017
    lbd = -1
    ubd = 0
    β = 0.953
    # τ = 0.15          # appears below too
    θ = 0.282
    # tol = 1e-10       # appears below too
    ϕ = 0.5
    δ = 0.8
    ρ = 0.9
    σ = 0.025
    τ = 0.5

    # B = zeros(Nb)     # useless command
    # Y = zeros(Ny)     # useless command

    #Initialize Bond grid
    minB = lbd
    maxB = ubd
    step = (maxB-minB) / (Nb-1)
    B = minB:step:maxB

    #Intitializing Endowment matrix
    σ_z = sqrt((σ^2)/(1-ρ^2))
    Step = 10*σ_z/(Ny-1)
    Y = -5*σ_z:Step:5*σ_z

    #Conditional Probability matrix
    P = zeros(Ny,Ny)
    tauchen(ρ, σ, Ny, P)

    #Utility function
    U(x) = x^(1-ϕ) / (1-ϕ)

    #Initialise arrays
    V = fill(1/((1-β)*(1-ϕ)),Ny, Nb)
    Price = fill(1/(1+rstar),Ny, Nb)
    Vr = zeros(Ny, Nb)
    Vd = zeros(Ny)
    decision = ones(Ny,Nb)

    err = 2000
    tol = 1e-6
    iter = 0


    #Initialize Shock grid
    sumdef = 0

    # time_vd = 0
    # time_vr = 0
    # time_decide = 0


    #3
    while (err > tol) & (iter < maxIter)
        V0 = deepcopy(V)
        Vd0 = deepcopy(Vd)
        Price0 = deepcopy(Price)
        prob = zeros(Ny, Nb)
        #display(V0)

        #5
        for ib in 1:Nb
            for iy = 1:Ny


                #compute default and repayment
                #7
                
                sumdef = U(exp((1-τ)*Y[iy]))
                for y in 1:Ny
                    sumdef += (β* P[iy,y]* (θ* V0[y,1] + (1-θ)* Vd0[y]))
                end
                Vd[iy] = sumdef

                #8

                Max = -Inf
                for b in 1:Nb
                    c = exp(Y[iy]) + B[ib] - Price0[iy,b]*B[b]
                    if c > 0
                        sumret = 0
                        for y in 1:Ny
                            sumret += P[iy,y]*V0[y,b]
                        end
                        vr = U(c) + β * sumret
                        Max = max(Max, vr)
                    end
                end
                Vr[iy,ib] = Max


                #Choose repay or default
                if (Vd[iy] < Vr[iy,ib])
                    V[iy,ib] = Vr[iy,ib]
                    decision[iy,ib] = 0
                else
                    V[iy,ib] = Vd[iy]
                    decision[iy,ib] = 1
                end

                #calculate debt price
                for y in 1:Ny
                    prob[iy,ib] += P[iy,y] * decision[y,ib]
                end
                Price[iy,ib] = (1-prob[iy,ib]) / (1+rstar)

            end
        end

        err = maximum(abs.(V-V0))
        PriceErr = maximum(abs.(Price-Price0))
        VdErr = maximum(abs.(Vd-Vd0))
        Vd = δ * Vd + (1-δ) * Vd0
        Price = δ * Price + (1-δ) * Price0
        V = δ * V + (1-δ) * V0
        iter = iter + 1
        if verbose
            println(@sprintf("Errors of round %.0f: Value error: %.2e, price error: %.2e, Vd error: %.2e", iter, err, PriceErr, VdErr))
        end

    end

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

    return Vr,Vd,decision,Price

end

##

@time VReturn, VDefault, Decision, Price = main_cpu(verbose=false, maxIter=250);

p_cpu = plot(title="cpu",[VReturn[:,50], VDefault])
gui(p_cpu)


##

@btime main();

#= Storing as CSV

using Parsers
using DataFrames
using CSV

dfPrice = DataFrame(Price)
dfVr = DataFrame(VReturn)
dfVd = DataFrame(VDefault)
dfDecision = DataFrame(Decision)

CSV.write("/Users/deng/Desktop/school/ECON8873/codes/Price.csv", dfPrice)
CSV.write("/Users/deng/Desktop/school/ECON8873/codes/Vr.csv", dfVr)
CSV.write("/Users/deng/Desktop/school/ECON8873/codes/Vd.csv", dfVd)
CSV.write("/Users/deng/Desktop/school/ECON8873/codes/Decision.csv", dfDecision)

=#