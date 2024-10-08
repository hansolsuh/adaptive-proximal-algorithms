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
using ProximalOperators: NormL1
using AdaProx

pgfplotsx()

struct LogisticLoss{TX,Ty}
    X::TX
    y::Ty
end

function AdaProx.eval_with_pullback(f::LogisticLoss, w)
    logits = f.X * w[1:end-1] .+ w[end]
    u = 1 .+ exp.(-logits)

    function logistic_loss_pullback()
        probs = 1 ./ u
        N = size(f.y, 1)
        grad = zero(w)
        grad[1:end-1] .= f.X' * (probs - f.y) ./ N
        grad[end] = mean(probs - f.y)
        return grad
    end

    return -mean((f.y .- 1) .* logits .- log.(u)), logistic_loss_pullback
end

(f::LogisticLoss)(w) = AdaProx.eval_with_pullback(f, w)[1]

function run_logreg_l1_data(
    filename,
    ::Type{T} = Float64;
    lam,
    tol = 1e-5,
    maxit = 1000,
) where {T}
   @info "Start L1 Logistic Regression ($filename)"

    X, y = load_libsvm_dataset(filename, T, labels = [0.0, 1.0])

    m, n = size(X)
    n = n + 1

    f = LogisticLoss(X, y)
    g = NormL1(T(lam))

    X1 = [X ones(m)]
    Lf = norm(X1 * X1') / 4 / m

    x0 = zeros(n)

    #comment out un-needed algorithm
    gam_init = 1.0 / Lf
    # run algorithm with 1/10 the tolerance to get "accurate" solution
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
#        run_logreg_l1_data(
#            joinpath(@__DIR__, "..", "datasets", "heart_scale"),
#            lam = 0.01, maxit = 2000, tol = 1e-7
#        )
    path = joinpath(@__DIR__, "mushrooms.jsonl")
    with_logger(get_logger(path)) do
        run_logreg_l1_data(
            joinpath(@__DIR__, "..", "datasets", "mushrooms"),
            lam = 0.01, maxit = 2000, tol = 1e-7
        )
    end
    plot_convergence(path)

    path = joinpath(@__DIR__, "heart_scale.jsonl")
    with_logger(get_logger(path)) do
        run_logreg_l1_data(
            joinpath(@__DIR__, "..", "datasets", "heart_scale"),
            lam = 0.01, maxit = 2000, tol = 1e-7
        )
    end
    plot_convergence(path)

    path = joinpath(@__DIR__, "heart_scale.jsonl")
        run_logreg_l1_data(
            joinpath(@__DIR__, "..", "datasets", "heart_scale"),
            lam = 0.01, maxit = 2000, tol = 1e-7
        )
    with_logger(get_logger(path)) do
        run_logreg_l1_data(
            joinpath(@__DIR__, "..", "datasets", "heart_scale"),
            lam = 0.01, maxit = 2000, tol = 1e-7
        )
    end
    plot_convergence(path)

    path = joinpath(@__DIR__, "phishing.jsonl")
    with_logger(get_logger(path)) do
        run_logreg_l1_data(
            joinpath(@__DIR__, "..", "datasets", "phishing"),
            lam = 0.01, maxit = 2000, tol = 1e-7
        )
    end
    plot_convergence(path)
end

#main()
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
