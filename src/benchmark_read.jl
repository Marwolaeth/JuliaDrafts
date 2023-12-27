using BenchmarkTools, Plots, StatsPlots, BenchmarkPlots

# Set WD
cd(raw"C:\Users\apavlyuchenko\Documents\Projects\Julia\ContentAnalysis")
path = "vectores/test_skipgram_1111_300.vec"
# readline(path)
include("readvec.jl")

# Warm-up
@time ReadVec.readvec(path)
@time ReadVec.readvec2(path)
@time ReadVec.read_vec(path)

bench = BenchmarkGroup([:small, :medium])
bench[:small] = BenchmarkGroup([:entire, :byline, :csvtable])
bench[:medium] = BenchmarkGroup([:entire, :byline, :csvtable])

# Benchmark on small Vec
bench[:small][:entire] = @benchmarkable ReadVec.readvec(path) samples = 50 seconds = 200 setup = (path = "vectores/test_skipgram_1111_300.vec")
bench[:small][:byline] = @benchmarkable ReadVec.readvec2(path) samples = 50 seconds = 200 setup = (path = "vectores/test_skipgram_1111_300.vec")
bench[:small][:csvtable] = @benchmarkable ReadVec.read_vec(path) samples = 50 seconds = 200 setup = (path = "vectores/test_skipgram_1111_300.vec")

# Benchmark on larger Vec
bench[:medium][:entire] = @benchmarkable ReadVec.readvec(path) samples = 50 seconds = 200 setup = (path = "vectores/test_skipgram_11111_300.vec")
bench[:medium][:byline] = @benchmarkable ReadVec.readvec2(path) samples = 50 seconds = 200 setup = (path = "vectores/test_skipgram_11111_300.vec")
bench[:medium][:csvtable] = @benchmarkable ReadVec.read_vec(path) samples = 50 seconds = 200 setup = (path = "vectores/test_skipgram_11111_300.vec")

results = run(bench, verbose=true)

# Plot (log y-axis)
plot(results[:small], yaxis=:log10, color=:royalblue, label=["Very small file" "" ""])
plot!(results[:medium], yaxis=:log10, color=:purple, label=["Larger file" "" ""], legend=true)

# Plot (standard y-axis)
plot(
    results[:small],
    color=:royalblue,
    label=["Very small file" "" ""],
    size=(1000, 750)
)
ylims!(0, 5.7e9)
plot!(
    results[:medium],
    color=:purple,
    label=["Larger file" "" ""],
    legend=true
)

BenchmarkTools.save("benchmarks/read_embeddings.json", results)

title!("Read embeddings benchmark results")
savefig("benchmarks/read_embeddings.png")

# A real test
path = "vectores/ruscorpora_upos_skipgram_300_5_2018.vec"
@time ReadVec.read_vec(path)