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

function test_fn(A, B)
    
    # a = threadIdx().x
    # b = threadIdx().y

    a = (blockIdx().x-1)*blockDim().x + threadIdx().x
    b = (blockIdx().y-1)*blockDim().y + threadIdx().y

    xstar, fstar = my_brent(x->a*x^2+b*x, Float32(-10), Float32(10))
    A[a,b] = xstar
    B[a,b] = fstar
    return nothing
end


##


n = 10
X = CUDA.zeros(n,n)
V = CUDA.zeros(n,n)

@cuda threads=(5,5) blocks=(2,2) test_fn(X,V)

X
V

B0 = (blockIdx().x-1)*blockDim().x + threadIdx().x
y0 = (blockIdx().y-1)*blockDim().y + threadIdx().y

##

lb = Float32(-2)
ub = Float32(2)
xstar, fstar = brent_test(x->x^2, lb, ub)