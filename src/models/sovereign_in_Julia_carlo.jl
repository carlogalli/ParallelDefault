# if pwd() != "/mnt/data/code/ParallelDefault"
#     cd("ParallelDefault")
# end
if pwd() != "/home/carlo/Documents/ParallelDefault"
    cd("/home/carlo/Documents/ParallelDefault")
end
using Pkg; Pkg.activate(".")

using Random, Distributions, Printf, BenchmarkTools, Plots, LinearAlgebra


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

struct ModelCPU
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

function ModelCPU(; Ny=21, Nb=100, rstar=0.017, lbd=-1, ubd=0, β=0.953, θ=0.282, ϕ=0.5, δ=0.8, ρ=0.9, σ=0.025, τ=0.5)

    return ModelCPU(Ny, Nb, rstar, lbd, ubd, β, θ, ϕ, δ, ρ, σ, τ)
end

#Utility function
U(m::ModelCPU, x::Real) = x^(1-m.ϕ) / (1-m.ϕ) 

function initModelCPU(m::ModelCPU)
    
    #Initializing Bond matrix
    minB = m.lbd; 
    maxB = m.ubd; 
    step = (maxB-minB) / (m.Nb-1)
    B = minB:step:maxB

    #Intitializing Endowment matrix
    σ_z = sqrt((m.σ^2)/(1-m.ρ^2))
    Step = 10*σ_z/(m.Ny-1)
    Y = -5*σ_z:Step:5*σ_z

    #Conditional Probability matrix
    P = zeros(m.Ny, m.Ny)
    tauchen(m.ρ, m.σ, m.Ny, P)    

    #Initialise arrays
    V = fill(1/((1-m.β)*(1-m.ϕ)), m.Ny, m.Nb)    
    Vr = zeros(m.Ny, m.Nb)
    Vd = zeros(m.Ny)
    Price = fill(1/(1+m.rstar), m.Ny, m.Nb)
    decision = ones(m.Ny, m.Nb)
    prob = zeros(m.Ny, m.Nb)

    return V, Vr, Vd, Price, decision, prob, P, Y, B
end

function bo_cpu(m::ModelCPU, V, Vr, Vd, Price, decision, prob, P, Y, B)

    #Default

    fv = (m.θ .* V[:,1] .+ (1-m.θ).* Vd)
    pv = @. (exp((1-m.τ)*Y)^(1-m.ϕ))/(1-m.ϕ)
    Vd .= pv + m.β * P * fv

    # @show pv

    for ib in 1:m.Nb
        for iy = 1:m.Ny


            #compute default and repayment
            
            # sumdef = U(m, exp((1-m.τ)*Y[iy]))
            # for y in 1:m.Ny
            #     sumdef += (m.β* P[iy,y]* (m.θ* V[y,1] + (1-m.θ)* Vd[y]))
            # end
            # Vd[iy] = sumdef

            Max = -Inf
            for b in 1:m.Nb
                c = exp(Y[iy]) + B[ib] - Price[iy,b]*B[b]
                if c > 0
                    sumret = 0
                    for y in 1:m.Ny
                        sumret += P[iy,y]*V[y,b]
                    end
                    vr = U(m, c) + m.β * sumret
                    Max = max(Max, vr)
                end
            end
            Vr[iy,ib] = Max
        end
    end

    #Choose repay or default
    @. decision_cpu = Vd < Vr
    @. V = decision * Vr + (1-decision) * Vd
    prob = P * decision
    @. Price = (1-prob)/(1+m.rstar)
end

function main_cpu(m::ModelCPU; verbose::Bool=false, maxIter::Int=300)

    V, Vr, Vd, Price, decision, prob, P, Y, B = initModelCPU(m)

    err = 2000
    tol = 1e-6
    iter = 0

    while (err > tol) & (iter < maxIter)
        V0 = deepcopy(V)
        Vd0 = deepcopy(Vd)
        Price0 = deepcopy(Price)

        bo_cpu(m, V, Vr, Vd, Price, decision, prob, P, Y, B)

        err = maximum(abs.(V-V0))        
        # Vd = δ * Vd + (1-δ) * Vd0
        # Price = δ * Price + (1-δ) * Price0
        # V = δ * V + (1-δ) * V0
        # iter = iter + 1
        if verbose
            PriceErr = maximum(abs.(Price-Price0))
            VdErr = maximum(abs.(Vd-Vd0))
            println(@sprintf("Errors of round %.0f: Value error: %.2e, price error: %.2e, Vd error: %.2e", iter, err, PriceErr, VdErr))
        end

    end

    println("Total Round ", iter, " error ", err)

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

mCPU = ModelCPU(Ny=21, Nb=100, rstar=0.017, lbd=-1, ubd=0, β=0.953, θ=0.282, ϕ=0.5, δ=0.8, ρ=0.9, σ=0.025, τ=0.5)

V_cpu, Vr_cpu, Vd_cpu, Price_cpu, decision_cpu, prob_cpu, P_cpu, Y_cpu, B_cpu = initModelCPU(mCPU)

bo_cpu(mCPU, V_cpu, Vr_cpu, Vd_cpu, Price_cpu, decision_cpu, prob_cpu, P_cpu, Y_cpu, B_cpu)


##

@btime bo_cpu(mCPU, V_cpu, Vr_cpu, Vd_cpu, Price_cpu, decision_cpu, prob_cpu, P_cpu, Y_cpu, B_cpu)
# 3.2 ms
# 13.7 ms with the broadcasting stuff


##
#= see if and where the values diverge =#

mCPU = ModelCPU(Ny=5, Nb=7, rstar=0.017, lbd=-1, ubd=0, β=0.953, θ=0.282, ϕ=0.5, δ=0.8, ρ=0.9, σ=0.025, τ=0.5)

V_cpu, Vr_cpu, Vd_cpu, Price_cpu, decision_cpu, prob_cpu, P_cpu, Y_cpu, B_cpu = initModelCPU(mCPU)

# Price_cpu
# B_cpu

bo_cpu(mCPU, V_cpu, Vr_cpu, Vd_cpu, Price_cpu, decision_cpu, prob_cpu, P_cpu, Y_cpu, B_cpu)

Vr_cpu



## debug Vr



if (Vd[iy] < Vr[iy,ib])
    V[iy,ib] = Vr[iy,ib]
    decision[iy,ib] = 0
else
    V[iy,ib] = Vd[iy]
    decision[iy,ib] = 1
end

#calculate debt price
for y in 1:m.Ny
    prob[iy,ib] += P[iy,y] * decision[y,ib]
end
Price[iy,ib] = (1-prob[iy,ib]) / (1+m.rstar)


@time 

##


m.β* P[iy,y]





for y in 1:m.Ny
    sumdef += (m.β* P[iy,y]* (m.θ* V[y,1] + (1-m.θ)* Vd[y]))
end


##

VReturn, VDefault, Decision, Price = main_cpu(verbose=false, maxIter=250);



##

@btime main();

#= see original code if you want to store stuff as CSV =#