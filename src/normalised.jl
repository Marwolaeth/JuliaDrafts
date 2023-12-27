using BenchmarkTools
using LinearAlgebra
using SparseArrays
using Distances
using StatsPlots
using BenchmarkPlots
include("cosine.jl")

A = [1 0 1 0; 0 1 0 1; 1 1 1 1; 0 0 0 0]
S = sparse(Float64.(A))
X = copy(S)

# This doesn't work
@inbounds @fastmath for i ∈ 1:n
    normalize!(S[i, :])
end

function cosineN(X::SparseMatrixCSC{Float64,Int64})::SparseMatrixCSC{Float64,Int64}
    n = size(X, 1)
    # Hard copy the transpose of X to process column-wise
    Xᵀ = copy(X')
    # Preallocate C
    C = spzeros(Float64, n, n)
    ## Norms
    @inbounds @fastmath for i ∈ 1:n
        Xᵀ[:, i] .= normalize(view(Xᵀ, :, i))
    end
    @inbounds @fastmath C .= Xᵀ' * Xᵀ
    return C
end

function cosineN(X::SparseMatrixCSC{Float32,Int64})::SparseMatrixCSC{Float32,Int64}
    n = size(X, 1)
    # Hard copy the transpose of X to process column-wise
    Xᵀ = copy(X')
    # Preallocate C
    C = spzeros(Float32, n, n)
    ## Norms
    @inbounds @fastmath for i ∈ 1:n
        Xᵀ[:, i] .= normalize(view(Xᵀ, :, i))
    end
    @inbounds @fastmath C .= Xᵀ' * Xᵀ
    return C
end

function cosineN(X::SparseMatrixCSC{Int64,Int64})::SparseMatrixCSC{Float32,Int64}
    n = size(X, 1)
    # Hard copy the transpose of X to process column-wise
    Xᵀ = Float32.(copy(X'))
    # Preallocate C
    C = spzeros(Float32, n, n)
    ## Norms
    @inbounds @fastmath for i ∈ 1:n
        Xᵀ[:, i] .= normalize(view(Xᵀ, :, i))
    end
    @inbounds @fastmath C .= Xᵀ' * Xᵀ
    return C
end

Cosine.cosine(S)
cosineN(S)
norm(Cosine.cosine(S) - cosineN(S))

cosineN(sparse(Float32.(A)))
norm(cosineN(sparse(Float32.(A))) - cosineN(S))

cosineN(sparse(A))

# BENCHMARK ----
dims = (1_000, 100_000, 0.01)
bench = BenchmarkGroup([:f32, :f64, :int])
bench[:f32] = BenchmarkGroup(["Divide", "Normalise"])
bench[:f64] = BenchmarkGroup(["Divide", "Normalise"])
bench[:int] = BenchmarkGroup(["Divide", "Normalise"])

bench[:f32]["Divide"] = @benchmarkable Cosine.cosine(S) samples = 100 setup = (S = sprand(Float32, dims...))
bench[:f32]["Normalise"] = @benchmarkable cosineN(S) samples = 100 setup = (S = sprand(Float32, dims...))
bench[:f64]["Divide"] = @benchmarkable Cosine.cosine(S) samples = 100 setup = (S = sprand(Float64, dims...))
bench[:f64]["Normalise"] = @benchmarkable cosineN(S) samples = 100 setup = (S = sprand(Float64, dims...))
bench[:int]["Divide"] = @benchmarkable Cosine.cosine(S) samples = 100 setup = (S = Int64.(sprand(Int16, dims...)))
bench[:int]["Normalise"] = @benchmarkable cosineN(S) samples = 100 setup = (S = Int64.(sprand(Int16, dims...)))

results = run(bench, verbose=true, seconds=180)
results[@tagged "Normalise"]
BenchmarkTools.save("benchmarks/sparse-test-N.json", results)

plot(
    results[:f32], yaxis=:log10,
    color=:royalblue, label=["" "Float32"], fα=0.5,
    size=(1000, 750)
)
plot!(
    results[:f64], yaxis=:log10,
    color=:purple, label=["" "Float64"], fα=0.5
)
plot!(
    results[:int], yaxis=:log10,
    color=:red3, label=["" "Integer"], legend=true, fα=0.5
)
title!("Cosine distances benchmark results:\n1k by 100k sparse matrix")
savefig("benchmarks/sparse-sl-1x.png")