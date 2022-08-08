
function SciMLBase.solve(prob::IntegralProblem, ::Nothing, sensealg, lb, ub, p, args...;
                         reltol = 1e-8, abstol = 1e-8, kwargs...)
    if lb isa Number
        __solve(prob, QuadGKJL(); reltol = reltol, abstol = abstol, kwargs...)
    elseif length(lb) > 8 && reltol < 1e-4 || abstol < 1e-4
        __solve(prob, VEGAS(); reltol = reltol, abstol = abstol, kwargs...)
    else
        __solve(prob, HCubatureJL(); reltol = reltol, abstol = abstol, kwargs...)
    end
end

function SciMLBase.solve(prob::IntegralProblem,
                         alg::SciMLBase.AbstractIntegralAlgorithm,
                         args...;
                         sensealg = ReCallVJP(ZygoteVJP()),
                         do_inf_transformation = nothing,
                         kwargs...)
    prob = transformation_if_inf(prob, do_inf_transformation)
    __solvebp(prob, alg, sensealg, prob.lb, prob.ub, prob.p, args...; kwargs...)
end

# Give a layer to intercept with AD
__solvebp(args...; kwargs...) = __solvebp_call(args...; kwargs...)

function __solvebp_call(prob::IntegralProblem, ::QuadGKJL, sensealg, lb, ub, p, args...;
                        reltol = 1e-8, abstol = 1e-8,
                        maxiters = typemax(Int),
                        kwargs...)
    if isinplace(prob) || lb isa AbstractArray || ub isa AbstractArray
        error("QuadGKJL only accepts one-dimensional quadrature problems.")
    end
    @assert prob.batch == 0
    @assert prob.nout == 1
    p = p
    f = x -> prob.f(x, p)
    val, err = quadgk(f, lb, ub,
                      rtol = reltol, atol = abstol,
                      kwargs...)
    SciMLBase.build_solution(prob, QuadGKJL(), val, err, retcode = :Success)
end

function __solvebp_call(prob::IntegralProblem,
                        ::HCubatureJL,
                        sensealg, lb, ub, p, args...;
                        reltol = 1e-8, abstol = 1e-8,
                        maxiters = typemax(Int),
                        kwargs...)
    p = p

    if isinplace(prob)
        dx = zeros(prob.nout)
        f = (x) -> (prob.f(dx, x, p); dx)
    else
        f = (x) -> prob.f(x, p)
    end
    @assert prob.batch == 0

    if lb isa Number
        val, err = hquadrature(f, lb, ub;
                               rtol = reltol, atol = abstol,
                               maxevals = maxiters, kwargs...)
    else
        val, err = hcubature(f, lb, ub;
                             rtol = reltol, atol = abstol,
                             maxevals = maxiters, kwargs...)
    end
    SciMLBase.build_solution(prob, HCubatureJL(), val, err, retcode = :Success)
end

function __solvebp_call(prob::IntegralProblem, alg::VEGAS,
                        sensealg, lb, ub, p, args...;
                        reltol = 1e-8, abstol = 1e-8,
                        maxiters = typemax(Int),
                        kwargs...)
    p = p
    @assert prob.nout == 1
    if prob.batch == 0
        if isinplace(prob)
            dx = zeros(prob.nout)
            f = (x) -> (prob.f(dx, x, p); dx)
        else
            f = (x) -> prob.f(x, p)
        end
    else
        if isinplace(prob)
            dx = zeros(prob.batch)
            f = (x) -> (prob.f(dx, x', p); dx)
        else
            f = (x) -> prob.f(x', p)
        end
    end
    val, err, chi = vegas(f, lb, ub, rtol = reltol, atol = abstol,
                          maxiter = maxiters, nbins = alg.nbins,
                          ncalls = alg.ncalls, batch = prob.batch != 0, kwargs...)
    SciMLBase.build_solution(prob, alg, val, err, chi = chi, retcode = :Success)
end

### Forward-Mode AD Intercepts

# Direct AD on solvers with QuadGK and HCubature
function __solvebp(prob, alg::QuadGKJL, sensealg, lb, ub,
                   p::AbstractArray{<:ForwardDiff.Dual{T, V, P}, N}, args...;
                   kwargs...) where {T, V, P, N}
    __solvebp_call(prob, alg, sensealg, lb, ub, p, args...; kwargs...)
end

function __solvebp(prob, alg::HCubatureJL, sensealg, lb, ub,
                   p::AbstractArray{<:ForwardDiff.Dual{T, V, P}, N}, args...;
                   kwargs...) where {T, V, P, N}
    __solvebp_call(prob, alg, sensealg, lb, ub, p, args...; kwargs...)
end

# Manually split for the pushforward
function __solvebp(prob, alg, sensealg, lb, ub,
                   p::AbstractArray{<:ForwardDiff.Dual{T, V, P}, N}, args...;
                   kwargs...) where {T, V, P, N}
    primal = __solvebp_call(prob, alg, sensealg, lb, ub, ForwardDiff.value.(p), args...;
                            kwargs...)

    nout = prob.nout * P

    if isinplace(prob)
        dfdp = function (out, x, p)
            dualp = reinterpret(ForwardDiff.Dual{T, V, P}, p)
            if prob.batch > 0
                dx = similar(dualp, prob.nout, size(x, 2))
            else
                dx = similar(dualp, prob.nout)
            end
            prob.f(dx, x, dualp)

            ys = reinterpret(ForwardDiff.Dual{T, V, P}, dx)
            idx = 0
            for y in ys
                for p in ForwardDiff.partials(y)
                    out[idx += 1] = p
                end
            end
            return out
        end
    else
        dfdp = function (x, p)
            dualp = reinterpret(ForwardDiff.Dual{T, V, P}, p)
            ys = prob.f(x, dualp)
            if prob.batch > 0
                out = similar(p, V, nout, size(x, 2))
            else
                out = similar(p, V, nout)
            end

            idx = 0
            for y in ys
                for p in ForwardDiff.partials(y)
                    out[idx += 1] = p
                end
            end

            return out
        end
    end
    rawp = copy(reinterpret(V, p))

    dp_prob = IntegralProblem(dfdp, lb, ub, rawp; nout = nout, batch = prob.batch,
                              kwargs...)
    dual = __solvebp_call(dp_prob, alg, sensealg, lb, ub, rawp, args...; kwargs...)
    res = similar(p, prob.nout)
    partials = reinterpret(typeof(first(res).partials), dual.u)
    for idx in eachindex(res)
        res[idx] = ForwardDiff.Dual{T, V, P}(primal.u[idx], partials[idx])
    end
    if primal.u isa Number
        res = first(res)
    end
    SciMLBase.build_solution(prob, alg, res, primal.resid)
end
