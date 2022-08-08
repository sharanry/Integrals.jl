function scale_x!(_x, ub, lb, x)
    _x .= (ub .- lb) .* x .+ lb
    _x
end

function scale_x(ub, lb, x)
    (ub .- lb) .* x .+ lb
end

function v_inf(t)
    return t ./ (1 .- t .^ 2)
end

function v_semiinf(t, a, upto_inf)
    if upto_inf == true
        return a .+ (t ./ (1 .- t))
    else
        return a .+ (t ./ (1 .+ t))
    end
end
