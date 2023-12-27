using BenchmarkTools
using SparseArrays
using Distances
using StatsPlots
using BenchmarkPlots
include("cosine.jl")

# Check ----
A = sprand(4, 6, 1 / 4)
Cosine.cosine(A)
1.0 .- pairwise(CosineDist(), A', A')

# A Single Matrix ----
dims = (1_000, 100_000, 0.1)
S = sprand(Float32, dims...)
C1 = Cosine.cosine(S)
C2 = pairwise(CosineDist(), S', S')
Float64(norm((1 .- C2) - C1))
(1 .- C2) - C1

bench = BenchmarkGroup([:f32, :f64, :int])
bench[:f32] = BenchmarkGroup(["CosineDist()", "cosine()"])
bench[:f64] = BenchmarkGroup(["CosineDist()", "cosine()"])
bench[:int] = BenchmarkGroup(["CosineDist()", "cosine()"])

bench[:f32]["CosineDist()"] = @benchmarkable pairwise(CosineDist(), S', S') samples = 100 setup = (S = sprand(Float32, dims...))
bench[:f32]["cosine()"] = @benchmarkable Cosine.cosine(S) samples = 100 setup = (S = sprand(Float32, dims...))
bench[:f64]["CosineDist()"] = @benchmarkable pairwise(CosineDist(), S', S') samples = 100 setup = (S = sprand(Float64, dims...))
bench[:f64]["cosine()"] = @benchmarkable Cosine.cosine(S) samples = 100 setup = (S = sprand(Float64, dims...))
bench[:int]["CosineDist()"] = @benchmarkable pairwise(CosineDist(), S', S') samples = 100 setup = (S = Int64.(sprand(Int16, dims...)))
bench[:int]["cosine()"] = @benchmarkable Cosine.cosine(S) samples = 100 setup = (S = Int64.(sprand(Int16, dims...)))

results = run(bench, verbose=true, seconds=180)
results[@tagged "cosine()"]
BenchmarkTools.save("benchmarks/sparse-sl-1x.json", results)

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

# Dense ----
dims = (100, 1_000)
A = rand(Float32, dims...)
C1 = Cosine.cosine(A)
C2 = pairwise(CosineDist(), A', A')
Float64(norm((1 .- C2) - C1))
(1 .- C2) - C1

bench = BenchmarkGroup([:f32, :f64])
bench[:f32] = BenchmarkGroup(["CosineDist()", "cosine()"])
bench[:f64] = BenchmarkGroup(["CosineDist()", "cosine()"])

bench[:f32]["CosineDist()"] = @benchmarkable pairwise(CosineDist(), A', A') samples = 300 setup = (A = rand(Float32, dims...))
bench[:f32]["cosine()"] = @benchmarkable Cosine.cosine(A) samples = 300 setup = (A = rand(Float32, dims...))
bench[:f64]["CosineDist()"] = @benchmarkable pairwise(CosineDist(), A', A') samples = 300 setup = (A = rand(Float64, dims...))
bench[:f64]["cosine()"] = @benchmarkable Cosine.cosine(A) samples = 300 setup = (A = rand(Float64, dims...))

results = run(bench, verbose=true, seconds=120)
results[@tagged "cosine()"]

plot(results[:f32], yaxis=:log10, color=:royalblue, label="Float32")
plot!(results[:f64], yaxis=:log10, color=:purple, label="Float64", legend=true)

# We suck with greater dense matrices