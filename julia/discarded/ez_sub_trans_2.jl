"""

Second transformation.


The subordinate ADP has policy operator

    (B_σ h)(w) = { Σ_e [ r(w, σ(w), e)^α + β h(σ(w)) ]^(1/θ) φ(e) }^θ 

The optimal policy is found by solving

    σ(w, e) = argmin_s { (w - s + e)^α + β * h(s) }

"""

include("ez_model.jl")
include("dp_code.jl")
include("plot_functions.jl")


"Action-value aggregator for the modified model."
function B(i, k, h::Vector, model)
    (; α, β, γ, θ, φ, e_grid, w_grid) = model
    w, s = w_grid[i], w_grid[k]
    value = Inf
    function f(e)
        return ((w - s + e)^α + β * h[k])^θ
    end
    if s <= w
        value = @views dot(f.(e_grid), φ)^(1/θ)
    end
    return value
end


model = create_ez_model()
(; α, β, γ, θ, φ, e_grid, w_grid) = model

println("Solving unmodified model.")
v_init = ones(length(model.w_grid), length(model.e_grid))
@time v_star, σ_star = optimistic_policy_iteration(v_init, B, model)

println("Solving modified model.")

h_init = ones(length(model.w_grid))
@time h_star, _ = optimistic_policy_iteration(h_init, B, model)

w_n, e_n = length(w_grid), length(e_grid)
σ_star_mod = Array{Int32}(undef, w_n, e_n)
function G_obj(i, j, k) 
    w, e, s = w_grid[i], e_grid[j], w_grid[k]
    value = Inf
    if s <= w
        value = ((w - s + e)^α + β * h_star[k])
    end
    return value
end

for i in 1:w_n
    for j in 1:e_n
        _, σ_star_mod[i, j] = findmin(G_obj(i, j, k) for k in 1:w_n)
    end
end


plot_policy(σ_star, model, title="original")
plot_policy(σ_star_mod, model, title="transformed")
plot_value_orig(v_star, model)
plot_value_mod(h_star, model)
