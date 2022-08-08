function transform_inf_number(t, p, f, lb, ub)
    if lb == -Inf && ub == Inf
        j = (1 .+ t .^ 2) / (1 .- t .^ 2) .^ 2
        return f(v_inf(t), p) * (j)
    elseif lb != -Inf && ub == Inf
        a = lb
        j = 1 ./ ((1 .- t) .^ 2)
        return f(v_semiinf(t, a, 1), p) * (j)
    elseif lb == -Inf && ub != Inf
        a = ub
        j = 1 ./ ((1 .+ t) .^ 2)
        return f(v_semiinf(t, a, 0), p) * (j)
    end
end

function transform_inf(t, p, f, lb, ub)
    if lb isa Number && ub isa Number
        return transform_inf_number(t, p, f, lb, ub)
    end

    lbb = lb .== -Inf
    ubb = ub .== Inf
    _none = .!lbb .& .!ubb
    _inf = lbb .& ubb
    semiup = .!lbb .& ubb
    semilw = lbb .& .!ubb

    function v(t)
        return t .* _none + v_inf(t) .* _inf + v_semiinf(t, lb, 1) .* semiup +
               v_semiinf(t, ub, 0) .* semilw
    end
    jac = ChainRulesCore.@ignore_derivatives ForwardDiff.jacobian(x -> v(x), t)
    j = det(jac)
    # j = 1.0
    f(v(t), p) * (j)
end

function transformation_if_inf(prob, ::Val{true})
    g = prob.f
    h(t, p) = transform_inf(t, p, g, prob.lb, prob.ub)
    if (prob.lb isa Number && prob.ub isa Number)
        if (prob.ub == Inf || prob.lb == -Inf)
            if prob.lb == -Inf && prob.ub == Inf
                lb = -1.00
                ub = 1.00
            elseif prob.lb != -Inf && prob.ub == Inf
                lb = 0.00
                ub = 1.00
            elseif prob.lb == -Inf && prob.ub != Inf
                lb = -1.00
                ub = 0.00
            end
        end
    elseif prob.lb isa AbstractVector && prob.ub isa AbstractVector
        if -Inf in prob.lb || Inf in prob.ub
            lbb = prob.lb .== -Inf
            ubb = prob.ub .== Inf
            _none = .!lbb .& .!ubb
            _inf = lbb .& ubb
            _semiup = .!lbb .& ubb
            _semilw = lbb .& .!ubb

            lb = 0.00 .* _semiup + -1.00 .* _inf + -1.00 .* _semilw + _none .* prob.lb
            ub = 1.00 .* _semiup + 1.00 .* _inf + 0.00 .* _semilw + _none .* prob.ub
        end
    end
    prob_ = remake(prob, f = h, lb = lb, ub = ub)
    return prob_
end

function transformation_if_inf(prob, ::Nothing)
    if (prob.lb isa Number && prob.ub isa Number && (prob.ub == Inf || prob.lb == -Inf)) ||
       -Inf in prob.lb || Inf in prob.ub
        return transformation_if_inf(prob, Val(true))
    end
    return prob
end

function transformation_if_inf(prob, ::Val{false})
    return prob
end

function transformation_if_inf(prob, do_inf_transformation = nothing)
    transformation_if_inf(prob, do_inf_transformation)
end
