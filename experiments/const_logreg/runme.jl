include(joinpath(@__DIR__, "..", "libsvm.jl"))
include(joinpath(@__DIR__, "..", "logging.jl"))

using Random
using LinearAlgebra
using Statistics
using Logging: with_logger
using Tables
using DataFrames
using Plots
using LaTeXStrings
using ProximalCore
using ProximalOperators: IndBallLinf
using AdaProx

pgfplotsx()

#Constrained logistic regression of Mai-Johansson

struct LogisticLoss2{TX,Ty,R}
    X::TX
    y::Ty
    c::R
end

# f(x) = (1/m)*sum_{i=1}^m log(1 + exp(-b_i*(ai'*x))) + 0.5*alpha*||x||^2 with ai in R^n, bi in R
function AdaProx.eval_with_pullback(f::LogisticLoss2, w)
    (m,n) = size(f.X)
    yX = f.X .* f.y
    yXw = yX * w

    loss = - yXw

    mask = yXw .> -50
    loss[mask] = log.(1. .+ exp.(-yXw[mask]))

    fval = sum(loss) / m + 0.5* f.c * (w'w)

    function logistic_loss_pullback()
        p = -1. ./ (1. .+ exp.(yXw))
        g = (yX'p) ./ (m + 0.5*f.c*(w'w))
        return g
    end

    return fval, logistic_loss_pullback
end

(f::LogisticLoss2)(w) = AdaProx.eval_with_pullback(f, w)[1]

function run_logreg_linf_data(
    filename,
    ::Type{T} = Float64;
    lam,
    tol = 1e-5,
    maxit = 1000,
) where {T}
   @info "Start L1 Logistic Regression ($filename)"

    X, y = load_libsvm_dataset(filename, T, labels = [-1.0, 1.0])

    m, n = size(X)

    f = LogisticLoss2(X, y, lam)
    g = IndBallLinf(T(1.))

    Lf = opnorm(Array(f.X))^2 / m / 4

    x0 = zeros(n)

    #comment out un-needed algorithm
    gam_init = 1.0 / Lf
    # run algorithm with 1/10 the tolerance to get "accurate" solution
    sol, numit = AdaProx.aapga_mj(
        x0,
        f = AdaProx.Counting(f),
        g = g,
        gamma = gam_init,
        aa_size = 5,
        tol = tol,
        maxit = maxit/2,
        name = "AA-PG-MJ"
    )
#    sol, numit = AdaProx.adapgm_my1(
#        x0,
#        f = f,
#        g = g,
#        rule = AdaProx.OurRule(gamma = gam_init),
#        tol = tol / 10,
#        maxit = maxit * 10,
#        name = nothing
#    )
#    sol, numit = AdaProx.adaptive_proxgrad(
#        x0,
#        f = f,
#        g = g,
#        rule = AdaProx.OurRule(gamma = gam_init),
#        tol = tol / 10,
#        maxit = maxit * 10,
#        name = nothing
#    )

    sol, numit = AdaProx.fixed_proxgrad(
        x0,
        f = AdaProx.Counting(f),
        g = g,
        gamma = gam_init,
        tol = tol,
        maxit = maxit,
        name = "PGM (1/Lf)"
    )

#    xi_values = [1.5, 2]
#    for xi = xi_values
#        sol, numit = AdaProx.backtracking_proxgrad(
#            zeros(n),
#            f = AdaProx.Counting(f),
#            g = g,
#            gamma0 = gam_init,
#            xi = xi, #increase in stepsize
#            tol = tol,
#            maxit = maxit/2,
#            name = "PGM (backtracking)-(xi=$(xi))"
#        )
#    end
#
#    sol, numit = AdaProx.backtracking_nesterov(
#        x0,
#        f = AdaProx.Counting(f),
#        g = g,
#        gamma0 = gam_init,
#        tol = tol,
#        maxit = maxit/2,
#        name = "Nesterov (backtracking)"
#    )
    sol, numit = AdaProx.fixed_nesterov(
        x0,
        f = AdaProx.Counting(f),
        g = g,
        gamma = gam_init,
        tol = tol,
        maxit = maxit/2,
        name = "Nesterov (fixed)"
    )

#    sol, numit = AdaProx.adaptive_proxgrad(
#        x0,
#        f = AdaProx.Counting(f),
#        g = g,
#        rule = AdaProx.MalitskyMishchenkoRule(gamma = gam_init),
#        tol = tol,
#        maxit = maxit,
#        name = "AdaPGM (MM)"
#    )
#
#    sol, numit = AdaProx.adaptive_proxgrad(
#        x0,
#        f = AdaProx.Counting(f),
#        g = g,
#        rule = AdaProx.OurRule(gamma = gam_init),
#        tol = tol,
#        maxit = maxit,
#        name = "AdaPGM (Ours)"
#    )
#
#    sol, numit = AdaProx.agraal(
#        x0,
#        f = AdaProx.Counting(f),
#        g = g,
#        gamma0 = gam_init,
#        tol = tol,
#        maxit = maxit,
#        name = "aGRAAL"
#    )
end

function plot_convergence(path)
    df = eachline(path) .|> JSON.parse |> Tables.dictrowtable |> DataFrame
    optimal_value = minimum(df[!, :objective])
    gb = groupby(df, :method)

    fig = plot(
        title = "Logistic regression ($(basename(path)))",
        xlabel = L"\mbox{call to } \mathcal A, \mathcal A'",
        ylabel = L"F(x^k) - F_\star",
    )

    for k in keys(gb)
        if k.method === nothing
            continue
        end
        plot!(
            # each evaluation of f is one mul with A
            # each gradient of f is one additional mul with A'
            gb[k][!, :grad_f_evals] + gb[k][!, :f_evals],
            max.(1e-14, gb[k][!, :objective] .- optimal_value),
            yaxis = :log,
            label = k.method,
        )
    end

    savefig(fig, joinpath(@__DIR__, "$(basename(path)).pdf"))
end

function main()

#        run_logreg_linf_data(
#            joinpath(@__DIR__, "..", "datasets", "madelon.t"),
#            lam = 0.1, maxit = 2000, tol = 1e-7
#        )

    path = joinpath(@__DIR__, "madelon.t.jsonl")
    with_logger(get_logger(path)) do
        run_logreg_linf_data(
            joinpath(@__DIR__, "..", "datasets", "madelon.t"),
            lam = 0.1, maxit = 2000, tol = 1e-7
        )
    end
    plot_convergence(path)
#below examples really dont work. dont feel like searching for lambda weight
#    path = joinpath(@__DIR__, "mushrooms.jsonl")
#    with_logger(get_logger(path)) do
#        run_logreg_linf_data(
#            joinpath(@__DIR__, "..", "datasets", "mushrooms"),
#            lam = 0.1, maxit = 2000, tol = 1e-7
#        )
#    end
#    plot_convergence(path)
#
#    path = joinpath(@__DIR__, "heart_scale.jsonl")
#    with_logger(get_logger(path)) do
#        run_logreg_linf_data(
#            joinpath(@__DIR__, "..", "datasets", "heart_scale"),
#            lam = 0.1, maxit = 2000, tol = 1e-7
#        )
#    end
#    plot_convergence(path)
#
#    path = joinpath(@__DIR__, "heart_scale.jsonl")
#        run_logreg_linf_data(
#            joinpath(@__DIR__, "..", "datasets", "heart_scale"),
#            lam = 0.1, maxit = 2000, tol = 1e-7
#        )
#    with_logger(get_logger(path)) do
#        run_logreg_linf_data(
#            joinpath(@__DIR__, "..", "datasets", "heart_scale"),
#            lam = 0.1, maxit = 2000, tol = 1e-7
#        )
#    end
#    plot_convergence(path)
#
#    path = joinpath(@__DIR__, "phishing.jsonl")
#    with_logger(get_logger(path)) do
#        run_logreg_linf_data(
#            joinpath(@__DIR__, "..", "datasets", "phishing"),
#            lam = 0.1, maxit = 2000, tol = 1e-7
#        )
#    end
#    plot_convergence(path)
end

#main()
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
