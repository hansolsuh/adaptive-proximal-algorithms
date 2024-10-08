module AdaProx

using Logging
using LinearAlgebra
using ProximalCore: prox, convex_conjugate, Zero

const Record = Logging.LogLevel(-1)

# Gradient evaluation interface

eval_with_pullback(f, _) = @error("eval_with_pullback not defined for type $(typeof(f))")

function eval_with_gradient(f, x)
    f_x, pb = eval_with_pullback(f, x)
    return f_x, pb()
end

# Utilities

include("./counting.jl")

is_logstep(n; base = 10) = mod(n, base^(log(base, n) |> floor)) == 0

nan_to_zero(v) = ifelse(isnan(v), zero(v), v)

upper_bound(x, f_x, grad_x, z, gamma) = f_x + real(dot(grad_x, z - x)) + 1 / (2 * gamma) * norm(z - x)^2

# Proximal-gradient methods with backtracking stepsize ("sufficient descent").
#
# See sections 10.4.2 and 10.7 from Amir Beck, "First-Order Methods in Optimization,"
# MOS-SIAM Series on Optimization, SIAM, 2017.
# https://my.siam.org/Store/Product/viewproduct/?ProductId=29044686

function backtrack_stepsize(gamma, f, g, x, f_x, grad_x, shrink=0.5, it=nothing)
    z, g_z = prox(g, x - gamma * grad_x, gamma)
    ub_z = upper_bound(x, f_x, grad_x, z, gamma)
    f_z, pb = eval_with_pullback(f, z)
    back_it = 0
    while f_z > ub_z
        gamma *= shrink
        if gamma < 1e-12
            @error "step size became too small ($gamma)"
        end
        z, g_z = prox(g, x - gamma * grad_x, gamma)
        ub_z = upper_bound(x, f_x, grad_x, z, gamma)
        f_z, pb = eval_with_pullback(f, z)
        back_it = back_it + 1
    end
#    println("backtracking iter : $(back_it), stepsize: $(gamma) ")
    return gamma, z, f_z, g_z, pb
end

function backtracking_proxgrad(x0; f, g, gamma0, xi = 1.0, shrink = 0.5, tol = 1e-5, maxit = 100_000, name = "Backtracking PG")
    x, z, gamma = x0, x0, gamma0
    f_x, grad_x = eval_with_gradient(f, x)
    for it = 1:maxit
        gamma, z, f_z, g_z, pb = backtrack_stepsize(xi * gamma, f, g, x, f_x, grad_x, shrink, it)
        norm_res = norm(z - x) / gamma
        @logmsg Record "" method=name it gamma norm_res objective=(f_z + g_z) grad_f_evals=grad_count(f) prox_g_evals=prox_count(g) f_evals=eval_count(f)
        if norm_res <= tol
            return z, it
        end
        x, f_x = z, f_z
        grad_x = pb()
#        println("backtracking proxgrad, iter: $(it), f_val: $(f_x), stepsize: $(gamma)")
    end
    return z, maxit
end

function backtracking_nesterov(x0; f, g, gamma0, shrink = 0.5, tol = 1e-5, maxit = 100_000, name = "Backtracking Nesterov")
    x, z, gamma = x0, x0, gamma0
    theta = one(gamma)
    f_x, grad_x = eval_with_gradient(f, x)
    for it = 1:maxit
        z_prev = z
        gamma, z, f_z, g_z, _ = backtrack_stepsize(gamma, f, g, x, f_x, grad_x, shrink, it)
        norm_res = norm(z - x) / gamma
        @logmsg Record "" method=name it gamma norm_res objective=(f_z + g_z) grad_f_evals=grad_count(f) prox_g_evals=prox_count(g) f_evals=eval_count(f)
        if norm_res <= tol
            return z, it
        end
        theta_prev = theta
        theta = (1 + sqrt(1 + 4 * theta_prev^2)) / 2
        x = z + (theta_prev - 1) / theta * (z - z_prev)
        f_x, grad_x = eval_with_gradient(f, x)
    end
    return z, maxit
end

# Fixed stepsize fast proximal gradient
#
# See Chambolle, Pock, "An introduction to continuous optimization for imaging," 
# Acta Numerica, 25 (2016), 161–319.

function fixed_nesterov(x0; f,g, Lf = nothing, muf = 0, mug = 0, gamma = nothing, theta = nothing, tol = 1e-5, maxit = 100_000, name = "Fixed Nesterov")
    @assert (gamma === nothing) != (Lf === nothing)
    if gamma === nothing
        gamma = 1 / Lf
    end
    mu = muf + mug
    q = gamma * mu / (1 + gamma * mug)
    @assert q < 1
    if theta === nothing
        theta = if q > 0
            1 / sqrt(q)
        else
            0
        end
    end
    @assert 0 <= theta <= 1 / sqrt(q)
    x, x_prev = x0, x0
    for it = 1:maxit
        theta_prev = theta
        if mu == 0
            theta = (1 + sqrt(1 + 4 * theta_prev^2)) / 2
            beta = (theta_prev - 1) / theta
        else
            theta = (1 - q * theta_prev^2 + sqrt((1 - q * theta_prev^2)^2 + 4 * theta_prev^2)) / 2
            beta = (theta_prev - 1) * (1 + gamma * mug - theta * gamma * mu) / theta / (1 - gamma * muf)
        end
        z = x + beta * (x - x_prev)
        _, grad_z = eval_with_gradient(f, z)
        x_prev = x
        x, g_x = prox(g, z - gamma * grad_z, gamma)
        norm_res = norm(x - z) / gamma
        without_counting() do
            @logmsg Record "" method=name it gamma norm_res objective=(f(x) + g_x) grad_f_evals=grad_count(f) prox_g_evals=prox_count(g) f_evals=eval_count(f)
        end
        if norm_res <= tol
            return x, it
        end
    end
    return x, maxit
end

#Nesterov, but follows Mai-Johansson formulation - (feasible set stuff)
function fixed_fista_aapga(x0; f,g, Lf = nothing, muf = 0, mug = 0, gamma = nothing, theta = nothing, tol = 1e-5, maxit = 100_000, name = "Fixed Nesterov a la AA-MJ formulation")
    @assert (gamma === nothing) != (Lf === nothing)
    if gamma === nothing
        gamma = 1 / Lf
    end
    mu = muf + mug
    q = gamma * mu / (1 + gamma * mug)
    @assert q < 1
    if theta === nothing
        theta = if q > 0
            1 / sqrt(q)
        else
            0
        end
    end
    @assert 0 <= theta <= 1 / sqrt(q)
    x, x_prev = x0, x0
    y, y_prev = x0, x0
    _, grad_x = eval_with_gradient(f, x)
    gvec = x - gamma * grad_x
    for it = 1:maxit

        x_prev = x
        x, g_x = prox(g, y, gamma)
        grad_prev = grad_x
        _, grad_x = eval_with_gradient(f, x)
        y_prev = y
        gvec_prev = gvec

        gvec = x - gamma * grad_x


        theta_prev = theta
        if mu == 0
            theta = (1 + sqrt(1 + 4 * theta_prev^2)) / 2
            beta = (theta_prev - 1) / theta
        else
            theta = (1 - q * theta_prev^2 + sqrt((1 - q * theta_prev^2)^2 + 4 * theta_prev^2)) / 2
            beta = (theta_prev - 1) * (1 + gamma * mug - theta * gamma * mu) / theta / (1 - gamma * muf)
        end
        y = gvec + beta * (gvec - gvec_prev)

        norm_res = norm(x - gvec) / gamma
        without_counting() do
            @logmsg Record "" method=name it gamma norm_res objective=(f(x) + g_x) grad_f_evals=grad_count(f) prox_g_evals=prox_count(g) f_evals=eval_count(f)
        end
        if norm_res <= tol
            return x, it
        end
    end
    return x, maxit
end

#Solve for alpha for AA
# Assumes R matrix is already truncated to be of right size
# Returns array of alpha
function aa_lsq(R, reg)
    RTR = R' * R
    R_norm = opnorm(RTR)
    RTR = RTR ./ R_norm
    n = size(RTR,2)
    b = ones(n)
    RTR = RTR + reg*I
    x = 0 #UNDEF error.... 
    try
        x = RTR \ b
    catch
        x = qr(RTR, Val(true)) \ b
    end
#    x = qr(RTR, Val(true)) \ b
    temp = sum(x)
    alpha = x / temp
    return alpha
end

# Takes matrix A,  appends vector x, and pops first col if colsize is > n
function aa_append_mat(A, x, n)
    A = isempty(A) ? x : hcat(A, x)
    # Discard old data
    if size(A,2) > n
        A = A[1:end, 2:end]
    end
    return A
end

#AA-PGA
function aapga_mj(x0; f,g, Lf = nothing, gamma = nothing, tol = 1e-5, aa_size = nothing, aa_reg = 1e-10, maxit = 10_000, name = "Anderson Accelerated Proximal Gradient by Mai-Johansson")
    @assert (gamma === nothing) != (Lf === nothing)
    if gamma === nothing
        gamma = 1 / Lf
    end

    # Default size 10
    if aa_size === nothing
        aa_size = 10
    end

    R = Array{Float64}(undef, 0, 0)
    G = Array{Float64}(undef, 0, 0)
    x, x_prev = x0, x0
    y, y_prev = x0, x0
    fx, grad_x = eval_with_gradient(f, x)

    y_ext = x
    gk = x - gamma*grad_x
    r = gk - y_ext
    G = aa_append_mat(G, gk, aa_size)
    R = aa_append_mat(R, r, aa_size)

    y_ext = gk
    x, _ = prox(g, y_ext, gamma) # Why negative??

    for it = 1:maxit
        mk = min(aa_size, it)
        fx, grad_x = eval_with_gradient(f, x)

        gk = x - gamma*grad_x
        r = gk - y_ext
        G = aa_append_mat(G, gk, aa_size)
        R = aa_append_mat(R, r, aa_size)

        x_prox, g_x =  prox(g, gk, gamma)
        grad_map = (x - x_prox) ./ gamma

        #Solve for alpha
        alpha = aa_lsq(R, aa_reg)
        #Extrapolate
        #Dirty... idk julia
        if (size(alpha,1) == 1)
            y_test =  G * alpha[1]
        else
            y_test =  G * alpha
        end
        x_test, g_x =  prox(g, y_test, gamma)

        # sufficient descent condition
        f_test, grad_test = eval_with_gradient(f, x_test)
        if f_test - fx <=  -0.5*gamma*(grad_map'grad_map)
            y_ext = y_test
            x = x_test
            fx = f_test
            grad_x = grad_test
        else
            y_ext = gk
            x = x_prox
            fx, grad_x = eval_with_gradient(f, x)
        end
        norm_res = norm(x - gk) / gamma
        without_counting() do
            @logmsg Record "" method=name it gamma norm_res objective=(f(x) + g_x) grad_f_evals=grad_count(f) prox_g_evals=prox_count(g) f_evals=eval_count(f)
        end
    end
    return x, maxit
end

# Adaptive Golden Ratio Algorithm.
#
# See Yura Malitsky, "Golden ratio algorithms for variational inequalities,"
# Mathematical Programming, Volume 184, Pages 383–410, 2020.
# https://link.springer.com/article/10.1007/s10107-019-01416-w

function agraal(
    x1;
    f,
    g,
    x0 = nothing,
    gamma0 = nothing,
    gamma_max = 1e6,
    phi = 1.5,
    tol = 1e-5,
    maxit = 100_000,
    name = "aGRAAL"
)
    if x0 === nothing
        x0 = x1 + randn(size(x1))
    end
    x, x_prev, x_bar = x1, x0, x1
    _, grad_x = eval_with_gradient(f, x)
    _, grad_x_prev = eval_with_gradient(f, x_prev)
    if gamma0 === nothing
        gamma0 = norm(x - x_prev) / norm(grad_x - grad_x_prev)
    end
    gamma = gamma0
    rho = 1 / phi + 1 / phi^2
    theta = one(gamma)
    for it = 1:maxit
        C = norm(x - x_prev)^2 / norm(grad_x - grad_x_prev)^2
        gamma_prev = gamma
        gamma = min(rho * gamma_prev, phi * theta * C / (4 * gamma_prev), gamma_max)
        theta = phi * gamma / gamma_prev
        x_bar = ((phi - 1) * x + x_bar) / phi
        x_prev, grad_x_prev = x, grad_x
        x, g_x = prox(g, x_bar - gamma * grad_x_prev, gamma)
        norm_res = norm(x - x_prev) / gamma
        without_counting() do
            @logmsg Record "" method=name it gamma norm_res objective=(f(x) + g_x) grad_f_evals=grad_count(f) prox_g_evals=prox_count(g) f_evals=eval_count(f)
        end
        if norm_res <= tol
            return x, it
        end
        _, grad_x = eval_with_gradient(f, x)
    end
    return x, maxit
end

# Fixed-step and adaptive primal-dual and proximal-gradient methods.
# All algorithms implemented as special cases of one generic loop.
#
# See:
# - Chapter 10 from Amir Beck, "First-Order Methods in Optimization,"
#   MOS-SIAM Series on Optimization, SIAM, 2017.
#   https://my.siam.org/Store/Product/viewproduct/?ProductId=29044686
# - Yura Malitsky, Konstantin Mishchenko "Adaptive Gradient Descent without Descent,"
#   Proceedings of the 37th International Conference on Machine Learning, PMLR 119:6702-6712, 2020.
#   https://proceedings.mlr.press/v119/malitsky20a.html
# - Laurent Condat, "A primal–dual splitting method for convex optimization involving Lipschitzian, proximable and linear composite terms,"
#   Journal of optimization theory and applications, Springer, 2013.
#   https://link.springer.com/article/10.1007/s10957-012-0245-9

Base.@kwdef struct FixedStepsize{R}
    gamma::R
    t::R = one(gamma)
end

function stepsize(rule::FixedStepsize, args...)
    return (rule.gamma, rule.gamma * rule.t^2), nothing
end

Base.@kwdef struct MalitskyMishchenkoRule{R}
    gamma::R
    t::R = one(gamma)
end

function stepsize(rule::MalitskyMishchenkoRule{R}) where {R}
    return (rule.gamma, rule.gamma * rule.t^2), (rule.gamma, R(Inf))
end

function stepsize(rule::MalitskyMishchenkoRule, (gamma_prev, rho), x1, grad_x1, x0, grad_x0)
    L = norm(grad_x1 - grad_x0) / norm(x1 - x0)
    gamma = min(sqrt(1 + rho) * gamma_prev, 1 / (2 * L))
    return (gamma, gamma * rule.t^2), (gamma, gamma / gamma_prev)
end


function adapgm_my1(x; f, g, rule, tol = 1e-5, maxit = 10_000, name = "MyAdaPGM")
    (gamma, sigma), state = stepsize(rule)

    _, grad_x = eval_with_gradient(f, x)
    v = x - gamma * (grad_x)
    x_prev, grad_x_prev = x, grad_x
    x, _ = prox(g, v, gamma)

    for it = 1:maxit
        f_x, grad_x = eval_with_gradient(f, x)

        primal_res = (v - x) / gamma + grad_x

        gamma_prev = gamma
        (gamma, sigma), state = stepsize(rule, state, x, grad_x, x_prev, grad_x_prev)

        norm_res = sqrt(norm(primal_res)^2)

        without_counting() do
            @logmsg Record "" method=name it gamma sigma norm_res objective=(f_x + g(x)) grad_f_evals=grad_count(f) prox_g_evals=prox_count(g) f_evals=eval_count(f)
        end

        if norm_res <= tol
            return x, it
        end

        v = x - gamma * (grad_x)
        x_prev, grad_x_prev = x, grad_x
        x, _ = prox(g, v, gamma)
    end
    return x, y, maxit
end


struct OurRulePlus{R}
    gamma::R
    xi::R
    nu::R
    r::R
end

function OurRulePlus(; gamma = 0, nu = 1, xi = 1, r = 1/2)
    _gamma = if gamma > 0
        gamma
    else
        error("you must provide gamma > 0")
    end
    R = typeof(_gamma)
    return OurRulePlus{R}(_gamma, R(xi), R(nu), R(r))
end

function stepsize(rule::OurRulePlus)
    gamma = rule.gamma
    return (gamma, gamma), (gamma, gamma)
end

function stepsize(rule::OurRulePlus, (gamma1, gamma0), x1, grad_x1, x0, grad_x0)
    C = norm(grad_x1 - grad_x0)^2 / dot(grad_x1 - grad_x0, x1 - x0) |> nan_to_zero
    L = dot(grad_x1 - grad_x0, x1 - x0) / norm(x1 - x0)^2 |> nan_to_zero
    D = 1- 2*rule.r + gamma1 * L * (gamma1 * C + 2*(rule.r-1) ) |> nan_to_zero
    gamma = gamma1 * min(
        sqrt( 1/(rule.r*(rule.nu + rule.xi)) + gamma1 / gamma0),
        sqrt( (rule.nu*(1+rule.xi) -1)/(rule.nu*(rule.nu+rule.xi)) ) / sqrt(max(D,0))
    )
    return (gamma, gamma), (gamma, gamma1)
end

struct OurRule{R}
    gamma::R
    t::R
    norm_A::R
    delta::R
    Theta::R
end

function OurRule(; gamma = 0, t = 1, norm_A = 0, delta = 0, Theta = 1.2)
    _gamma = if gamma > 0
        gamma
    elseif norm_A > 0
        1 / (2 * Theta * t * norm_A)
    else
        error("you must provide gamma > 0 if norm_A = 0")
    end
    R = typeof(_gamma)
    return OurRule{R}(_gamma, R(t), R(norm_A), R(delta), R(Theta))
end

function stepsize(rule::OurRule)
    gamma = rule.gamma
    sigma = rule.gamma * rule.t^2
    return (gamma, sigma), (gamma, gamma)
end

function stepsize(rule::OurRule, (gamma1, gamma0), x1, grad_x1, x0, grad_x0)
    xi = rule.t^2 * gamma1^2 * rule.norm_A^2
    C = norm(grad_x1 - grad_x0)^2 / dot(grad_x1 - grad_x0, x1 - x0) |> nan_to_zero
    L = dot(grad_x1 - grad_x0, x1 - x0) / norm(x1 - x0)^2 |> nan_to_zero
    D = gamma1 * L * (gamma1 * C - 1)
    gamma = min(
        gamma1 * sqrt(1 + gamma1 / gamma0),
        1 / (2 * rule.Theta * rule.t * rule.norm_A),
        (
            gamma1 * sqrt(1 - 4 * xi * (1 + rule.delta)^2) /
            sqrt(2 * (1 + rule.delta) * (D + sqrt(D^2 + xi * (1 - 4 * xi * (1 + rule.delta)^2))))
        ),
    )
    sigma = gamma * rule.t^2
    return (gamma, sigma), (gamma, gamma1)
end

function adaptive_primal_dual_my(x,y; f, g, h, A, rule, tol = 1e-5, maxit = 10_000, name = "AdaPDMMy",)
    (gamma, sigma), state = stepsize(rule)
    h_conj = convex_conjugate(h)

    A_x = A * x
    _, grad_x = eval_with_gradient(f, x)
    At_y = A' * y
    v = x - gamma * (grad_x + At_y)
    x_prev, A_x_prev, grad_x_prev = x, A_x, grad_x
    x, _ = prox(g, v, gamma)

    for it = 1:maxit
        A_x = A * x
        f_x, grad_x = eval_with_gradient(f, x)

        primal_res = (v - x) / gamma + grad_x + At_y

        gamma_prev = gamma
        (gamma, sigma), state = stepsize(rule, state, x, grad_x, x_prev, grad_x_prev)
        rho = gamma / gamma_prev

        w = y + sigma * ((1 + rho) * A_x - rho * A_x_prev)
        y, _ = prox(h_conj, w, sigma)

        dual_res = (w - y) / sigma - A_x
        norm_res = sqrt(norm(primal_res)^2 + norm(dual_res)^2)

        without_counting() do
            @logmsg Record "" method=name it gamma sigma norm_res objective=(f_x + g(x) + h(A_x)) grad_f_evals=grad_count(f) prox_g_evals=prox_count(g) prox_h_evals=prox_count(h) A_evals=mul_count(A) At_evals=amul_count(A) f_evals=eval_count(f)
        end

        if norm_res <= tol
            return x, y, it
        end

        At_y = A' * y
        v = x - gamma * (grad_x + At_y)
        x_prev, A_x_prev, grad_x_prev = x, A_x, grad_x
        x, _ = prox(g, v, gamma)
    end
    return x, y, maxit
end

function adaptive_linesearch_primal_dual_my(x, y; f, g, h, A, gamma = nothing, eta = 1.0, t = 1.0, delta = 1e-8, Theta = 1.2, r = 2, R = 0.95, tol = 1e-5, maxit = 10_000, name = "AdaPDM+", )
    @assert eta > 0 "eta must be positive"
    @assert Theta > (delta + 1) "must be Theta > (delta + 1)"

    if gamma === nothing
        gamma = 1 / (2 * Theta * t * eta)
    end

    f_val = 0
    g_val = 0
    h_val = 0
    @assert gamma <= 1 / (2 * Theta * t * eta) "gamma is too large"

    delta1 = 1 + delta
    gamma_prev = gamma
    h_conj = convex_conjugate(h)

    A_x = A * x
    f_val, grad_x = eval_with_gradient(f, x)
    At_y = A' * y
    v = x - gamma * (grad_x + At_y)
    x_prev, A_x_prev, grad_x_prev = x, A_x, grad_x
    x, g_val = prox(g, v, gamma)

    for it = 1:maxit
        A_x = A * x
        f_x, grad_x = eval_with_gradient(f, x)

        primal_res = (v - x) / gamma + grad_x + At_y

        C = norm(grad_x - grad_x_prev)^2 / dot(grad_x - grad_x_prev, x - x_prev) |> nan_to_zero
        L = dot(grad_x - grad_x_prev, x - x_prev) / norm(x - x_prev)^2 |> nan_to_zero
        Delta = gamma * L * (gamma * C - 1)
        xi_bar = t^2 * gamma^2 * eta^2 * delta1^2
        m4xim1 = (1 - 4 * xi_bar)

        eta = R * eta
        w = y
        sigma = t^2 * gamma
        ls_iter = 0
        while true
            gamma_next = min(
                gamma * sqrt(1 + gamma / gamma_prev),
                1 / (2 * Theta * t * eta),
                gamma * sqrt(m4xim1 / (2 * delta1 * (Delta + sqrt(Delta^2 + m4xim1 * (t * eta * gamma)^2)))),
            )
            rho = gamma_next / gamma
            sigma = t^2 * gamma_next
            w = y + sigma * ((1 + rho) * A_x - rho * A_x_prev)
            y_next, h_val = prox(h_conj, w, sigma)
            At_y_next = A' * y_next
            if eta >= norm(At_y_next - At_y) / norm(y_next - y)
                gamma, gamma_prev = gamma_next, gamma
                y, At_y = y_next, At_y_next
                break
            end
            eta *= r
            ls_iter = ls_iter + 1
        end

        dual_res = (w - y) / sigma - A_x
        norm_res = sqrt(norm(primal_res)^2 + norm(dual_res)^2)

        without_counting() do
            @logmsg Record "" method=name it gamma sigma norm_res objective=(f_x + g(x) + h(A_x)) grad_f_evals=grad_count(f) prox_g_evals=prox_count(g) prox_h_evals=prox_count(h) A_evals=mul_count(A) At_evals=amul_count(A) f_evals=eval_count(f)
        end
        if norm_res <= tol
            return x, y, it
        end

        funcval = f_val+g_val+h_val
        println("Iter : $(it), Function value: $(funcval), LS iter : $(ls_iter), residual: $(norm_res)")

        v = x - gamma * (grad_x + At_y)
        x_prev, A_x_prev, grad_x_prev = x, A_x, grad_x
        x, g_val = prox(g, v, gamma)
    end
    println("norm_res: $(norm_res)")
    return x, y, maxit
end


function adaptive_primal_dual(
    x,
    y;
    f,
    g,
    h,
    A,
    rule,
    tol = 1e-5,
    maxit = 10_000,
    name = "AdaPDM",
)
    (gamma, sigma), state = stepsize(rule)
    h_conj = convex_conjugate(h)

    A_x = A * x
    _, grad_x = eval_with_gradient(f, x)
    At_y = A' * y
    v = x - gamma * (grad_x + At_y)
    x_prev, A_x_prev, grad_x_prev = x, A_x, grad_x
    x, _ = prox(g, v, gamma)

    for it = 1:maxit
        A_x = A * x
        f_x, grad_x = eval_with_gradient(f, x)

        primal_res = (v - x) / gamma + grad_x + At_y

        gamma_prev = gamma
        (gamma, sigma), state = stepsize(rule, state, x, grad_x, x_prev, grad_x_prev)
        rho = gamma / gamma_prev

        w = y + sigma * ((1 + rho) * A_x - rho * A_x_prev)
        y, _ = prox(h_conj, w, sigma)

        dual_res = (w - y) / sigma - A_x
        norm_res = sqrt(norm(primal_res)^2 + norm(dual_res)^2)

        without_counting() do
            @logmsg Record "" method=name it gamma sigma norm_res objective=(f_x + g(x) + h(A_x)) grad_f_evals=grad_count(f) prox_g_evals=prox_count(g) prox_h_evals=prox_count(h) A_evals=mul_count(A) At_evals=amul_count(A) f_evals=eval_count(f)
        end

        if norm_res <= tol
            return x, y, it
        end

        At_y = A' * y
        v = x - gamma * (grad_x + At_y)
        x_prev, A_x_prev, grad_x_prev = x, A_x, grad_x
        x, _ = prox(g, v, gamma)
    end
    return x, y, maxit
end


function condat_vu(
    x,
    y;
    f,
    g,
    h,
    A,
    Lf,
    gamma = nothing,
    sigma = nothing,
    norm_A = nothing,
    tol = 1e-5,
    maxit = 10_000,
    name = "Condat-Vu",
)
    # # NOTE: Peviously I had parameter selection as per [Thm 3.1, Condat 2013]
    # # Implemented as follows (rho is relaxation parameter)
    #     if gamma === nothing && sigma !== nothing
    #         gamma = 0.99 / (Lf / 2 + sigma * norm_A^2)
    #     elseif gamma !== nothing && sigma === nothing
    #         sigma = 0.99 * (1 / gamma - Lf / 2) / norm_A^2
    #     end
    #     @assert gamma !== nothing && sigma !== nothing
    #     if rho === nothing
    #         delta = 2 - Lf / (2 * (1 / gamma - sigma * norm_A^2))
    #         rho = delta / 2
    #     end
    #     gamma_sigma = 1 / gamma - sigma * norm_A^2
    #     @assert gamma_sigma >= Lf / 2
    #     @assert (rho > 0) && (rho < 2 - Lf / 2 / gamma_sigma)

    if gamma === sigma === nothing
        R = typeof(Lf)
        par = R(5) # scaling parameter for comparing Lipschitz constants and \|L\|
        par2 = R(100)   # scaling parameter for α
        if norm_A === nothing
            norm_A = norm(A)
        end
        if norm_A > par * Lf
            alpha = R(1)
        else
            alpha = par2 * norm_A / Lf
        end
        gamma = R(1) / (Lf / 2 + norm_A / alpha)
        sigma = R(0.99) / (norm_A * alpha)
    end
    @assert gamma !== nothing && sigma !== nothing
    rule = FixedStepsize(gamma, sqrt(sigma / gamma))
    return adaptive_primal_dual(x, y; f, g, h, A, rule, tol, maxit, name)
end

function adaptive_proxgrad(x; f, g, rule, tol = 1e-5, maxit = 100_000, name = "AdaPGM")
    x, _, numit = adaptive_primal_dual(x, zero(x); f, g, h = Zero(), A = 0, rule, tol, maxit, name)
    return x, numit
end

function fixed_pgm_my(x; f, g, gamma, tol = 1e-5, maxit = 10_000, name = "MyFixedPGM")
    rule = FixedStepsize(gamma, one(gamma))
    (gamma, sigma), state = stepsize(rule)

    _, grad_x = eval_with_gradient(f, x)
    v = x - gamma * (grad_x)
    x_prev, grad_x_prev = x, grad_x
    x, _ = prox(g, v, gamma)

    for it = 1:maxit
        f_x, grad_x = eval_with_gradient(f, x)

        primal_res = (v - x) / gamma + grad_x

        gamma_prev = gamma

        norm_res = sqrt(norm(primal_res)^2)

        without_counting() do
            @logmsg Record "" method=name it gamma sigma norm_res objective=(f_x + g(x)) grad_f_evals=grad_count(f) prox_g_evals=prox_count(g) f_evals=eval_count(f)
        end

        if norm_res <= tol
            return x, it
        end

        v = x - gamma * (grad_x)
        x_prev, grad_x_prev = x, grad_x
        x, _ = prox(g, v, gamma)
    end
    return x, maxit
end


function auto_adaptive_proxgrad(x; f, g, gamma = nothing, tol = 1e-5, maxit = 100_000, name = "AutoAdaPGM")
    _, grad_x = eval_with_gradient(f, x)

    if norm(grad_x) <= tol
        return x, 0
    end

    if gamma === nothing 
        xeps = prox(x .- 0.1 * grad_x, 0.1) # proxgrad
        _, grad_xeps = eval_with_gradient(f, xeps)
        L = dot(grad_x - grad_xeps, x - xeps) / norm(x - xeps)^2
        gamma = iszero(L) ? 1.0 : 1 / L  
    end

    @assert gamma > 0

    x_prev, grad_x_prev, gamma_prev = x, grad_x, gamma
    x, _ = prox(g, x - gamma * grad_x, gamma)
    _, grad_x = eval_with_gradient(f, x)
    L = dot(grad_x - grad_x_prev, x - x_prev) / norm(x - x_prev)^2
    gamma = iszero(L) ? sqrt(2) * gamma : 1 / L

    if gamma_prev / gamma > 1e5  # if the inital guess was too large
        x, _ = prox(g, x_prev - gamma * grad_x_prev, gamma)
        _, grad_x = eval_with_gradient(f, x)
        L = dot(grad_x - grad_x_prev, x - x_prev) / norm(x - x_prev)^2
        gamma = iszero(L) ? sqrt(2) * gamma : 1 / L
    end

    rule = OurRule(; gamma, t=1, norm_A=0, delta=0, Theta=1.2)

    return adaptive_proxgrad(x_prev; f, g, rule, tol, maxit, name = name)
end

function fixed_proxgrad(x; f, g, gamma, tol = 1e-5, maxit = 100_000, name = "Fixed stepsize PGM")
    adaptive_proxgrad(x; f, g, rule = FixedStepsize(gamma, one(gamma)), tol, maxit, name)
end

# Linesearch version of adaptive_primal_dual ("fully adaptive")

function adaptive_linesearch_primal_dual(
    x,
    y;
    f,
    g,
    h,
    A,
    gamma = nothing,
    eta = 1.0,
    t = 1.0,
    delta = 1e-8,
    Theta = 1.2,
    r = 2,
    R = 0.95,
    tol = 1e-5,
    maxit = 10_000,
    name = "AdaPDM+",
)
    @assert eta > 0 "eta must be positive"
    @assert Theta > (delta + 1) "must be Theta > (delta + 1)"

    if gamma === nothing
        gamma = 1 / (2 * Theta * t * eta)
    end

    @assert gamma <= 1 / (2 * Theta * t * eta) "gamma is too large"

    delta1 = 1 + delta
    gamma_prev = gamma
    h_conj = convex_conjugate(h)

    A_x = A * x
    _, grad_x = eval_with_gradient(f, x)
    At_y = A' * y
    v = x - gamma * (grad_x + At_y)
    x_prev, A_x_prev, grad_x_prev = x, A_x, grad_x
    x, _ = prox(g, v, gamma)

    for it = 1:maxit
        A_x = A * x
        f_x, grad_x = eval_with_gradient(f, x)

        primal_res = (v - x) / gamma + grad_x + At_y

        C = norm(grad_x - grad_x_prev)^2 / dot(grad_x - grad_x_prev, x - x_prev) |> nan_to_zero
        L = dot(grad_x - grad_x_prev, x - x_prev) / norm(x - x_prev)^2 |> nan_to_zero
        Delta = gamma * L * (gamma * C - 1)
        xi_bar = t^2 * gamma^2 * eta^2 * delta1^2
        m4xim1 = (1 - 4 * xi_bar)

        eta = R * eta
        w = y
        sigma = t^2 * gamma
        ls_it = 0
        while true
            gamma_next = min(
                gamma * sqrt(1 + gamma / gamma_prev),
                1 / (2 * Theta * t * eta),
                gamma * sqrt(m4xim1 / (2 * delta1 * (Delta + sqrt(Delta^2 + m4xim1 * (t * eta * gamma)^2)))),
            )
            rho = gamma_next / gamma
            sigma = t^2 * gamma_next
            w = y + sigma * ((1 + rho) * A_x - rho * A_x_prev)
            y_next, _ = prox(h_conj, w, sigma)
            At_y_next = A' * y_next
            if eta >= norm(At_y_next - At_y) / norm(y_next - y)
                gamma, gamma_prev = gamma_next, gamma
                y, At_y = y_next, At_y_next
                println("iter: $(it), ls iter: $(ls_it)")
                break
            end
            eta *= r
            ls_it = ls_it + 1
        end

        dual_res = (w - y) / sigma - A_x
        norm_res = sqrt(norm(primal_res)^2 + norm(dual_res)^2)

        without_counting() do
            @logmsg Record "" method=name it gamma sigma norm_res objective=(f_x + g(x) + h(A_x)) grad_f_evals=grad_count(f) prox_g_evals=prox_count(g) prox_h_evals=prox_count(h) A_evals=mul_count(A) At_evals=amul_count(A) f_evals=eval_count(f)
        end
        if norm_res <= tol
            return x, y, it
        end

        v = x - gamma * (grad_x + At_y)
        x_prev, A_x_prev, grad_x_prev = x, A_x, grad_x
        x, _ = prox(g, v, gamma)
    end
    return x, y, maxit
end

# Algorithm 4 of ``A first-order primal-dual algorithm with linesearch''
# (applied to the dual for consistency)

function backtrack_stepsize_MP(sigma, sigma_prev, t, x_prev, y, y_prev, grad_x_prev, A_x_prev, At_y, At_y_prev, f, g, A, f_x_prev)
    theta = sigma / sigma_prev
    gamma = t^2 * sigma
    At_ybar = (1+theta) * At_y - theta* At_y_prev
    v = x_prev - gamma * (At_ybar + grad_x_prev)
    x, _ = prox(g, v, gamma)
    A_x = A * x
    f_x, pb = eval_with_pullback(f, x)
    lhs = gamma * sigma * norm(A_x - A_x_prev)^2 + 2 * gamma * (f_x - f_x_prev - dot(grad_x_prev, x - x_prev))
    while lhs > 0.95 * norm(x - x_prev)^2
        sigma /= 2
        if sigma < 1e-12
            @error "step size became too small ($gamma)"
        end
        theta = sigma / sigma_prev
        gamma = t^2 * sigma
        At_ybar = (1+theta) * At_y - theta* At_y_prev
        v = x_prev - gamma * (At_ybar + grad_x_prev)
        x, _ = prox(g, v, gamma)
        A_x = A * x
        f_x, pb = eval_with_pullback(f, x)
        lhs = gamma * sigma * norm(A_x - A_x_prev)^2 + 2 * gamma * (f_x - f_x_prev - dot(grad_x_prev, x - x_prev))
    end
    return sigma, gamma, x, v, A_x, f_x, pb
end

function malitsky_pock(
    x,
    y;
    f,
    g,
    h,
    A,
    sigma,
    t = 1.0, # t = gamma / sigma > 0
    tol = 1e-5,
    maxit = 10_000,
    name = "MP-ls",
)
    h_conj = convex_conjugate(h)
    theta = one(sigma)
    y_prev = y
    A_x = A * x
    At_y = A' * y
    for it = 1:maxit
        At_y_prev = At_y 
        w = y + sigma * A_x
        y, _ = prox(h_conj, w, sigma)
        At_y = A' * y

        sigma_prev = sigma
        sigma = sigma * sqrt(1 + theta)

        f_x_prev, grad_x_prev = eval_with_gradient(f, x)
        x_prev, A_x_prev = x, A_x
        sigma, gamma, x, v, A_x, f_x, pb =
            backtrack_stepsize_MP(sigma, sigma_prev, t, x_prev, y, y_prev, grad_x_prev, A_x_prev, At_y, At_y_prev, f, g, A, f_x_prev)
        grad_x = pb()

        y_prev = y

        primal_res = (v - x) / gamma + grad_x + At_y
        dual_res = (w - y) / sigma_prev - A_x
        norm_res = sqrt(norm(primal_res)^2 + norm(dual_res)^2)

        without_counting() do
            @logmsg Record "" method=name it gamma sigma norm_res objective=(f_x + g(x) + h(A_x)) grad_f_evals=grad_count(f) prox_g_evals=prox_count(g) prox_h_evals=prox_count(h) A_evals=mul_count(A) At_evals=amul_count(A) f_evals=eval_count(f)
        end

        if norm_res <= tol
            return x, y, it
        end
    end
    return x, y, maxit
end

end
