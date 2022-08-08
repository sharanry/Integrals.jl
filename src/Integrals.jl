module Integrals

using Reexport, MonteCarloIntegration, QuadGK, HCubature
@reexport using SciMLBase
using Zygote, ReverseDiff, ForwardDiff, LinearAlgebra

import ChainRulesCore
import ChainRulesCore: NoTangent
import ZygoteRules

struct QuadGKJL <: SciMLBase.AbstractIntegralAlgorithm end

struct HCubatureJL <: SciMLBase.AbstractIntegralAlgorithm end

struct VEGAS <: SciMLBase.AbstractIntegralAlgorithm
    nbins::Int
    ncalls::Int
end

VEGAS(; nbins = 100, ncalls = 1000) = VEGAS(nbins, ncalls)

abstract type QuadSensitivityAlg end
struct ReCallVJP{V}
    vjp::V
end

abstract type IntegralVJP end
struct ZygoteVJP end
struct ReverseDiffVJP
    compile::Bool
end
export QuadGKJL, HCubatureJL, VEGAS

include("utils.jl")
include("transform.jl")
include("solve.jl")
include("ad.jl")

end # module
