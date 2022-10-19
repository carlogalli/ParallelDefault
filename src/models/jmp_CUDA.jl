if pwd() != "/mnt/data/code/ParallelDefault"
    cd("ParallelDefault")
end
using Pkg; Pkg.activate(".")

using Random, Distributions, CUDA, Printf, BenchmarkTools

struct Model_GPU
    # Model Parameters
    β::Float32      # df govt
    β_hh::Float32   # df hh
    σ::Float32      # risk aversion
    α_m::Float32    # MIU scaling factor
    η::Float32      # MIU curvature
    γ::Float32      # govt spending utility curvature
    g_lb::Float32   # govt spending lower bound
    α_g::Float32    # govt spending weight
    r::Float32      # intnl risk-free rate
    ρ::Float32      # income process persistence
    σ_ϵ::Float32    # income process error standard deviation
    θ::Float32      # ree-enter probability
    h::Float32      # haircut on debt
    d0::Float32     # linear coefficient on default costadd C
    d1::Float32     # quadratic coefficient on default cost
    ρ_B::Float32    # coefficient of Gumbel debt taste shocks
    ρ_μ::Float32    # coefficient of Gumbel mu_D taste shocks
    ρ_δ::Float32    # coefficient of Gumbel enforcement shocks
    λ::Float32      # debt maturity
    κ::Float32      # debt coupon

    ny::Int32
    nB::Int32
    N::Int32

    Bgrid_lb::Float32
    Bgrid_ub::Float32
end

function Model_GPU(;
    β=.83, β_hh=.99, σ=2., α_m=2e-5, η=3., γ=2., g_lb=0., α_g=0.074, r=0.00598,
    ρ=0.9293, σ_ϵ=0.0115, θ=0.282, h=0.37, d0=-0.4, d1=0.44, 
    ρ_B=1e-3, ρ_μ=1e-3, ρ_δ=1e-3, λ=0.0465, κ=0.,
    ny=51, nB=150, Bgrid_lb=1e-2, Bgrid_ub=2.5
    )

    N = ny*nB

    return Model_GPU(β, β_hh, σ, α_m, η, γ, g_lb, α_g, r, ρ, σ_ϵ, θ, h, d0, d1, ρ_B, ρ_μ, ρ_δ, λ, κ, ny, nB, N, Bgrid_lb, Bgrid_ub)
end

function tauchen_carlo(N::Int32, ρ::Float32, σ_ϵ::Float32; μ::Real=0, n_std::Real=3)

    @show N

    # process (x_t above) standard deviation (std is ϵ's standard deviation)
    σ_y = σ_ϵ/sqrt(1-ρ^2)
    grid_bar = n_std * σ_y      # grid bounds
    grid = collect(Float32, range(μ-grid_bar, stop=μ+grid_bar, length=N))
    dif = grid[2]-grid[1]
  
    # Get transition probabilities
    P = zeros(Float32, N, N)
    
    # return nothing

    for row in 1:N
        
        # do endpoints first
        P[row, 1] = cdf(Normal(0., 1.), (grid[1]+dif/2 -(1-ρ)*μ -ρ*grid[row])/σ_ϵ )
        P[row, N] = 1-cdf(Normal(0., 1.), (grid[N]-dif/2 -(1-ρ)*μ -ρ*grid[row])/σ_ϵ )

        # middle columns
        for col in 2:N-1
            P[row, col] = cdf(Normal(0., 1.), (grid[col]+dif/2 -(1-ρ)*μ -ρ*grid[row])/σ_ϵ ) - cdf(Normal(0., 1.), (grid[col]-dif/2 -(1-ρ)*μ -ρ*grid[row])/σ_ϵ )
        end
    end

    # @show P
    # normalize
    sums = sum(P, dims=2)    

    P = @. P/sums
    
    if maximum(abs.(sum(P, dims=2).-1)) > 1e-6
        error("Matrix rows must sum up to 1!")
    end
  
    return grid, P
end

function ydef_fn(m::Model_GPU, y::Float32)
    return y - max(0, m.d0*y + m.d1*y^2)
end

#Utility function
function u_fn(x, σ, α)
    # if σ!=1
    #     return α * (max(x, 1e-14)^(1-σ))/(1-σ)
    # elseif σ==1
    #     return α * log(max(x, 1e-14))
    # end
    return ifelse(
        σ!=1, 
        α*(max(x, 1e-14)^(1-σ))/(1-σ), 
        α*log(max(x, 1e-14))
        )
end

function U_fn(m, c, rb)
    return u_fn(c, m.σ, 1.) + u_fn(rb, m.η, m.α_m)
end

function my_brent(
    f::Function , x_lower::Float32, x_upper::Float32;
    rel_tol::Float32 = sqrt(eps(Float32)),
    abs_tol::Float32 = eps(Float32),
    iterations::Int32 = Int32(1000),
    my_trace::Bool = false)


    if x_lower > x_upper
        error("x_lower must be less than x_upper")
    end

    golden_ratio = Float32(1)/2 * (3 - sqrt(Float32(5.0)))

    new_minimizer = x_lower + golden_ratio*(x_upper-x_lower)
    new_minimum = f(new_minimizer)
    best_bound = "initial"
    f_calls = 1 # Number of calls to f
    step = zero(Float32)
    old_step = zero(Float32)

    old_minimizer = new_minimizer
    old_old_minimizer = new_minimizer

    old_minimum = new_minimum
    old_old_minimum = new_minimum

    iteration = 0
    converged = false
    stopped_by_callback = false

    while iteration < iterations && !stopped_by_callback

        p = zero(Float32)
        q = zero(Float32)

        x_tol = rel_tol * abs(new_minimizer) + abs_tol

        x_midpoint = (x_upper+x_lower)/2

        if abs(new_minimizer - x_midpoint) <= 2*x_tol - (x_upper-x_lower)/2
            converged = true
            break
        end

        iteration += 1

        if abs(old_step) > x_tol
            # Compute parabola interpolation
            # new_minimizer + p/q is the optimum of the parabola
            # Also, q is guaranteed to be positive

            r = (new_minimizer - old_minimizer) * (new_minimum - old_old_minimum)
            q = (new_minimizer - old_old_minimizer) * (new_minimum - old_minimum)
            p = (new_minimizer - old_old_minimizer) * q - (new_minimizer - old_minimizer) * r
            q = 2(q - r)

            if q > 0
                p = -p
            else
                q = -q
            end
        end

        if abs(p) < abs(q*old_step/2) && p < q*(x_upper-new_minimizer) && p < q*(new_minimizer-x_lower)
            old_step = step
            step = p/q

            # The function must not be evaluated too close to x_upper or x_lower
            x_temp = new_minimizer + step
            if ((x_temp - x_lower) < 2*x_tol || (x_upper - x_temp) < 2*x_tol)
                step = (new_minimizer < x_midpoint) ? x_tol : -x_tol
            end
        else
            old_step = (new_minimizer < x_midpoint) ? x_upper - new_minimizer : x_lower - new_minimizer
            step = golden_ratio * old_step
        end

        # The function must not be evaluated too close to new_minimizer
        if abs(step) >= x_tol
            new_x = new_minimizer + step
        else
            new_x = new_minimizer + ((step > 0) ? x_tol : -x_tol)
        end

        new_f = f(new_x)
        f_calls += 1

        if new_f < new_minimum
            if new_x < new_minimizer
                x_upper = new_minimizer
                best_bound = "upper"
            else
                x_lower = new_minimizer
                best_bound = "lower"
            end
            old_old_minimizer = old_minimizer
            old_old_minimum = old_minimum
            old_minimizer = new_minimizer
            old_minimum = new_minimum
            new_minimizer = new_x
            new_minimum = new_f
        else
            if new_x < new_minimizer
                x_lower = new_x
            else
                x_upper = new_x
            end
            if new_f <= old_minimum || old_minimizer == new_minimizer
                old_old_minimizer = old_minimizer
                old_old_minimum = old_minimum
                old_minimizer = new_x
                old_minimum = new_f
            elseif new_f <= old_old_minimum || old_old_minimizer == new_minimizer || old_old_minimizer == old_minimizer
                old_old_minimizer = new_x
                old_old_minimum = new_f
            end
        end
    end

    if my_trace
        out = iteration, iteration == iterations, converged, rel_tol, abs_tol, f_calls
        return new_minimizer, new_minimum, out
    else
        return new_minimizer, new_minimum
    end
end


m = Model_GPU(nB=10, ny=10)

# Bond grid
Bgrid = collect(range(m.Bgrid_lb, stop=m.Bgrid_ub, length=m.nB))
Bgrid = CuArray(Bgrid)

# Endowment grid and transition probs
grid, P = tauchen_carlo(m.ny, m.ρ, m.σ_ϵ, n_std=3, μ=0.)
ygrid=CuArray(exp.(grid))
P = CuArray(P)

# Initialise arrays
v0, v, ev, evd, def_policy = [CUDA.zeros(m.nB, m.ny) for i in 1:5]
vr, c_rep, rb_rep, Bprime_rep, μ_rep, i_rep, q_rep, qtilde_rep, moneyEulerRHS_rep, bondsEulerRHS_rep = [CUDA.zeros(m.nB, m.ny) for i in 1:10]
vd, c_def, rb_def, Bprime_def, μ_def, i_def, q_def, qtilde_def, moneyEulerRHS_def, bondsEulerRHS_def = [CUDA.zeros(m.nB, m.ny) for i in 1:10]
q0_rep = CUDA.zeros(m.nB, m.ny)
vs, cs, rbs, μs, qs, is, cps = [CUDA.zeros(m.nB) for i in 1:7]


# Collect arrays in tuples
grids = (Bgrid, ygrid, P)
arrays_rep = (vr, c_rep, rb_rep, Bprime_rep, μ_rep, i_rep, q_rep, qtilde_rep)
arrays_def = (vd, c_def, rb_def, Bprime_def, μ_def, i_def, q_def, qtilde_def)
arrays = (def_policy, v, ev, evd)
eulers_rep = (moneyEulerRHS_rep, bondsEulerRHS_rep)
eulers_def = (moneyEulerRHS_def, bondsEulerRHS_def)
arrays_temp = (vs, cs, rbs, μs, qs, is, cps)


# Bgrid, ygrid, P = grids
# vr, c_rep, rb_rep, Bprime_rep, μ_rep, i_rep, q_rep, qtilde_rep = arrays_rep
# vd, c_def, rb_def, Bprime_def, μ_def, i_def, q_def, qtilde_def = arrays_def



function model_init!(m, grids, arrays_rep, arrays_def)
    Bgrid, ygrid, ~ = grids
    vr, c_rep, rb_rep, Bprime_rep, ~, ~, q_rep, ~ = arrays_rep
    vd, c_def, rb_def, Bprime_def, μ_def, ~, q_def, ~ = arrays_def

    B0 = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y0 = (blockIdx().y-1)*blockDim().y + threadIdx().y
    # tid_x, tid_y, bid_x, bid_y = threadIdx().x, threadIdx().y, blockIdx().x, blockIdx().y
    # @cuprintln("y0 $y0, B0 $B0, tid_x $tid_x, tid_y $tid_y, bid_x $bid_x, bid_y $bid_y")

    if B0 <= length(Bgrid) && y0 <= length(ygrid)
        # #= Repayment: in the last period, max U(c,rb,g) s.t. c=y-̃B*rb =#
        rb_rep[B0,y0] = 0.5 * ygrid[y0]/Bgrid[B0]
        c_rep[B0,y0] = ygrid[y0]-Bgrid[B0]*rb_rep[B0,y0]
        vr[B0,y0] = (c_rep[B0,y0]^(1-m.σ))/(1-m.σ) + m.α_m*(rb_rep[B0,y0]^(1-m.η))/(1-m.η)
        Bprime_rep[B0,y0] = 0.
        q_rep[B0, y0] = 1.
        
        # #= Default: in the last period, max U(c,rb,g) s.t. c=y_def-̃B*(1-h)*rb =#
        yd = ydef_fn(m,ygrid[y0])
        rb_def[B0,y0] = 0.5*yd/(Bgrid[B0]*(1-m.h))
        c_def[B0,y0] = yd-Bgrid[B0]*(1-m.h)*rb_def[B0,y0]
        vd[B0,y0] = (c_def[B0,y0]^(1-m.σ))/(1-m.σ) + m.α_m*(rb_def[B0,y0]^(1-m.η))/(1-m.η)
        # vd[B0,y0] = U_fn(m, c_def[B0,y0], rb_def[B0,y0])
        Bprime_def[B0,y0] = 0.
        μ_def[B0,y0] = 0.
        q_def[B0, y0] = 1.
    end

    return nothing
end

@cuda threads=(5,5) blocks=(2,2) model_init!(m, grids, arrays_rep, arrays_def)
# CUDA.synchronize()


function update_values_expectations!(
        m::Model_GPU, grids, arrays, arrays_rep, arrays_def, eulers_rep, eulers_def
    )

    Bgrid, ygrid, P = grids
    def_policy, v, ev, evd = arrays
    vr, c_rep, rb_rep, Bprime_rep, μ_rep, i_rep, q_rep, qtilde_rep = arrays_rep
    vd, c_def, rb_def, Bprime_def, μ_def, i_def, q_def, qtilde_def = arrays_def
    moneyEulerRHS_rep, bondsEulerRHS_rep = eulers_rep
    moneyEulerRHS_def, bondsEulerRHS_def = eulers_def

    B0 = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y0 = (blockIdx().y-1)*blockDim().y + threadIdx().y
    B1 = B0
    B1_def = B1     # to keep things clearer when computing qtilde_def
    B1_def_hc = CUDA.min(CUDA.searchsortedfirst(Bgrid, Bgrid[B1_def]*(1-m.h)), m.nB)
    
    # tid_x, tid_y, bid_x, bid_y = threadIdx().x, threadIdx().y, blockIdx().x, blockIdx().y
    # @cuprintln("y0 $y0, B0 $B0, B1_def_hc $B1_def_hc, tid_x $tid_x, tid_y $tid_y, bid_x $bid_x, bid_y $bid_y")

    if exp((vd[B0,y0]-vr[B0,y0])/m.ρ_δ) == Inf
        def_policy[B0,y0] = 1.
    else
        def_policy[B0,y0] = exp((vd[B0,y0]-vr[B0,y0])/m.ρ_δ) / (1+exp((vd[B0,y0]-vr[B0,y0])/m.ρ_δ))
    end    
    v[B0,y0] = def_policy[B0,y0]*vd[B0,y0] + (1-def_policy[B0,y0])*vr[B0,y0]

    # Reset values to zero
    ev[B1, y0], evd[B1, y0] = 0., 0.
    qtilde_rep[B1, y0], qtilde_def[B1, y0] = 0., 0.
    moneyEulerRHS_rep[B1, y0], moneyEulerRHS_def[B1, y0] = 0., 0.
    bondsEulerRHS_rep[B1, y0], bondsEulerRHS_def[B1, y0] = 0., 0.

    for y1 in 1:m.ny

        # compute expected value functions, makes them functions of (̃B',y)
        ev[B1, y0] += v[B1, y1] * P[y0, y1]
        evd[B1, y0] += vd[B1, y1] * P[y0, y1]

        #= 
        the update of the qtilde price fns should only use q_rep and q_def, not qtilde's 
        =#

        # uses: def_policy, q_rep, rb_rep, q_def, rb_def
        qtilde_rep[B1,y0] += (
            (1-def_policy[B1,y1])*(m.λ+(1-m.λ)*(m.κ+q_rep[B1,y1]))*rb_rep[B1,y1] + def_policy[B1,y1]*q_def[B1,y1]*rb_def[B1,y1]
        ) * P[y0,y1]
    
        # uses: q_def, rb_def here, q_rep, rb_rep below
        qtilde_def[B1_def, y0] += (
            (1-m.θ)*q_def[B1_def,y1]*rb_def[B1_def,y1] + 
            m.θ*(
                (1-def_policy[B1_def_hc,y1])*(m.λ+(1-m.λ)*(m.κ+q_rep[B1_def_hc,y1]))*rb_rep[B1_def_hc,y1] 
                + def_policy[B1_def_hc,y1]*q_def[B1_def_hc,y1]*rb_def[B1_def_hc,y1]
            )
        ) * P[y0,y1]

        # uses: def_policy, c_rep, rb_rep, c_def, rb_def
        moneyEulerRHS_rep[B1, y0] += m.β_hh * (
                (1-def_policy[B1,y1])*(c_rep[B1,y1]^(-m.σ) + m.α_m*rb_rep[B1,y1]^(-m.η))*rb_rep[B1,y1] + def_policy[B1,y1]*(c_def[B1,y1]^(-m.σ) + m.α_m*rb_def[B1,y1]^(-m.η))*rb_def[B1,y1]
            ) * P[y0,y1]

        # uses: def_policy, c_rep, rb_rep, c_def, rb_def
        moneyEulerRHS_def[B1, y0] += m.β_hh * (
                (1-m.θ)*(c_def[B1,y1]^(-m.σ) + m.α_m*rb_def[B1,y1]^(-m.η))*rb_def[B1,y1] + 
                m.θ*(1-def_policy[B1_def_hc,y1])*(c_rep[B1_def_hc,y1]^(-m.σ) + m.α_m*rb_rep[B1_def_hc,y1]^(-m.η))*rb_rep[B1_def_hc,y1] + 
                m.θ*def_policy[B1_def_hc,y1]*(c_def[B1_def_hc,y1]^(-m.σ) + m.α_m*rb_def[B1_def_hc,y1]^(-m.η))*rb_def[B1_def_hc,y1]
            ) * P[y0,y1]

        # uses: def_policy, c_rep, rb_rep, c_def, rb_def
        bondsEulerRHS_rep[B1, y0] += m.β_hh * (
                (1-def_policy[B1,y1])*(c_rep[B1,y1]^(-m.σ))*rb_rep[B1,y1] + 
                def_policy[B1,y1]*(c_def[B1,y1]^(-m.σ))*rb_def[B1,y1]
            ) * P[y0,y1]

        # uses: def_policy, c_rep, rb_rep, c_def, rb_def
        bondsEulerRHS_def[B1, y0] += m.β_hh * (
                (1-m.θ)*(c_def[B1,y1]^(-m.σ))*rb_def[B1,y1] + 
                m.θ*(1-def_policy[B1_def_hc,y1])*(c_rep[B1_def_hc,y1]^(-m.σ))*rb_rep[B1_def_hc,y1] + 
                m.θ*def_policy[B1_def_hc,y1]*(c_def[B1_def_hc,y1]^(-m.σ))*rb_def[B1_def_hc,y1]
            ) * P[y0,y1]
    end

    return nothing
end

@time @cuda threads=(5,5) blocks=(2,2) update_values_expectations!(m, grids, arrays, arrays_rep, arrays_def, eulers_rep, eulers_def)
# CUDA.synchronize()



function my_dot_product(xvec, yvec)
    temp = Float32(0)
    for i in 1:lastindex(xvec)
        temp += xvec[i]*yvec[i]
    end
    return temp
end

function vr_gridsearch(m::Model_GPU, grids, arrays, arrays_rep, eulers_rep, arrays_temp)

    Bgrid, ygrid, P = grids
    def_policy, v, ev, evd = arrays
    vr, c_rep, rb_rep, Bprime_rep, μ_rep, i_rep, q_rep, qtilde_rep = arrays_rep
    moneyEulerRHS_rep, bondsEulerRHS_rep = eulers_rep
    vs, cs, rbs, μs, qs, is, cps = arrays_temp
    vstar = Float32(-Inf)

    B0 = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y0 = (blockIdx().y-1)*blockDim().y + threadIdx().y

    tid_x, tid_y, bid_x, bid_y = threadIdx().x, threadIdx().y, blockIdx().x, blockIdx().y
    @cuprintln("y0 $y0, B0 $B0, tid_x $tid_x, tid_y $tid_y, bid_x $bid_x, bid_y $bid_y")
    
    # xstar, fstar = my_brent(x->B0*x^2+y0*x, Float32(-10), Float32(10))
    # vs[B0,y0] = fstar
    # cs[B0,y0] = xstar

    if B0 <= length(Bgrid) && y0 <= length(ygrid)
        
        for B1 in 1:m.nB

            # c, ~ = my_brent(
            #     c -> -U_fn(
            #             m, c, 
            #             (ygrid[y0]+qtilde_rep[B1,y0]*Bgrid[B1]-c)/(
            #                 Bgrid[B0]*(
            #                     m.λ+(1-m.λ)*m.κ + qtilde_rep[B1,y0]*(1-m.λ)*c^(-m.σ)/(
            #                             m.β_hh*moneyEulerRHS_rep[B1, y0]
            #                         )
            #                 )
            #             )
            #         )
            #     ,
            #     Float32(1e-6), ygrid[y0] + qtilde_rep[B1, y0]*Bgrid[B1]
            #     )
            
            c, ~ = my_brent(
                c -> -(c^(1-m.σ))/(1-m.σ) - m.α_m*((
                        (ygrid[y0]+qtilde_rep[B1,y0]*Bgrid[B1]-c)/(Bgrid[B0]*(m.λ+(1-m.λ)*m.κ + qtilde_rep[B1,y0]*(1-m.λ)*c^(-m.σ)/(m.β_hh*moneyEulerRHS_rep[B1, y0])))
                    )^(1-m.η))/(1-m.η), 
                    Float32(1e-4), ygrid[y0] + qtilde_rep[B1, y0]*Bgrid[B1]
                )

            rb = (ygrid[y0]+qtilde_rep[B1,y0]*Bgrid[B1]-c)/(
                Bgrid[B0]*(
                    m.λ+(1-m.λ)*m.κ + qtilde_rep[B1,y0]*(1-m.λ)*c^(-m.σ)/(
                            m.β_hh*moneyEulerRHS_rep[B1, y0]
                        )
                )
            )

            ucprime = c^(-m.σ)
            μ = moneyEulerRHS_rep[B1,y0] / (rb * ucprime) - 1

            cs[B1] = c
            μs[B1] = μ
            rbs[B1] = rb
            qs[B1] = qtilde_rep[B1, y0] / (rb*(1+μ)*(1+m.r))
            is[B1] = ((1+μ)*rb*ucprime) / bondsEulerRHS_rep[B1, y0]
            v = (c^(1-m.σ))/(1-m.σ) + m.α_m*(rb^(1-m.η))/(1-m.η) + m.β * ev[B1, y0]
            vs[B1] = v

            vstar = ifelse(vs[B1]>vstar, vs[B1], vstar)
            # @cuprintln(vstar)
        end

        D = 0.
        for i in 1:m.nB
            D += exp((vs[i]-vstar)/m.ρ_μ)
        end
        for i in 1:m.nB
            cps[i] = exp((vs[i]-vstar)/m.ρ_μ)/D
        end

        vr[B0,y0] = vstar + m.ρ_μ*log(D)
        
        # c_rep[B0,y0] = my_dot_product(cps, cs)
        # rb_rep[B0,y0] = my_dot_product(cps, rbs)
        # Bprime_rep[B0,y0] = my_dot_product(cps, Bgrid)
        # q_rep[B0,y0] = my_dot_product(cps, qs)
        # i_rep[B0,y0] = my_dot_product(cps, is)
        # μ_rep[B0,y0] = my_dot_product(cps, μs)

        c_rep[B0,y0], rb_rep[B0,y0], Bprime_rep[B0,y0], q_rep[B0,y0], i_rep[B0,y0], μ_rep[B0,y0] = 0., 0., 0., 0., 0., 0.
        for B1 in 1:m.nB
            c_rep[B0,y0] += cps[B1]*cs[B1]
            rb_rep[B0,y0] += cps[B1]*rbs[B1]
            Bprime_rep[B0,y0] += cps[B1]*Bgrid[B1]
            q_rep[B0,y0] += cps[B1]*qs[B1]
            i_rep[B0,y0] += cps[B1]*is[B1]
            μ_rep[B0,y0] += cps[B1]*μs[B1]
        end

    end

    return nothing
end


# @cuda threads=(1,1) blocks=(1,1) vr_gridsearch(m, grids, arrays, arrays_rep, eulers_rep, arrays_temp)
# CUDA.synchronize()


@cuda threads=(5,5) blocks=(2,2) vr_gridsearch(m, grids, arrays, arrays_rep, eulers_rep, arrays_temp)


# RESULTS VARY WITH THE EXECUTIONS!!!


##

nr = Int32(10^3)
nc = 100
A = rand(Float32, nr,nc);
B = rand(Float32, nr,nc);
A = CuArray(A)
B = CuArray(B)
C = CUDA.zeros(Float32, nc)

function f1!(X,Y,Z,nr)
    ic = threadIdx().x
    for j in 1:nr
        Z[ic] += X[j,ic]*Y[j,ic]
    end
    return nothing
end

function mydotproduct(X, Y)
    temp = Float32(0)
    for i in 1:100
        temp += X[i]*Y[i]
    end
    return temp
end

@time mydotproduct(A[:,1], B[:,1])

function f2!(X,Y,Z)
    ic = threadIdx().x
    # for j in 1:nr
    #     Z[ic] += X[j,ic]*Y[j,ic]
    # end
    Z[ic] = mydotproduct(X[:,ic], Y[:,ic])
    return nothing
end

@cuda threads=100 f2!(A,B,C)


##

function f3(X, Y, Z)
    i = threadIdx().x
    for j in 1:100
        Z[i] += X[j,i] * Y[j,i]
    end

    return nothing
end

g3(X,Y,Z) = @cuda threads=10 f3(X,Y,Z)

function my_dot_product(xvec, yvec)
    temp = Float32(0)
    for i in 1:lastindex(xvec)
        temp += xvec[i]*yvec[i]
    end
    return temp
end

function f4(X, Y, Z)
    i = threadIdx().x
    # xvec = view(X,:,i)
    # yvec = view(Y,:,i)
    # temp = mydotproduct(xvec, yvec)
    # Z[i] = temp
    Z[i] = my_dot_product(view(X,:,1), view(Y,:,1))

    return nothing
end

g4(X,Y,Z) = @cuda threads=10 f4(X, Y, Z)

g4(A,B,C4)

nr = Int32(10^2)
nc = 10
A = rand(Float32, nr,nc);
B = rand(Float32, nr,nc);
A = CuArray(A)
B = CuArray(B)
C3 = CUDA.zeros(Float32, nc)
C4 = CUDA.zeros(Float32, nc)
@btime g3($A, $B, $C3)
@btime g4($A, $B, $C4)

##



#= this is how to check number of threads and blocks =#
kernel = @cuda launch=false model_init!(arrays_rep)
config = launch_configuration(kernel.fun)
config.threads
config.blocks