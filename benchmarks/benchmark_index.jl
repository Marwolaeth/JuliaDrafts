using StatsBase
using BenchmarkTools, Plots, StatsPlots, BenchmarkPlots

# Set WD
cd(raw"ContentAnalysis")
path = "vectores/test_skipgram_1111_300.vec" # or maybe
path = "vectores/ruscorpora_upos_skipgram_300_5_2018.vec"
# readline(path)
include("readvec.jl")
using .ReadVec

# Warm-up & Check
@time embedding = read_vec(path)
embedding.embeddings
@time embedding2 = IndexedWordEmbedding(embedding);

embedding2.dict
@time embedding3 = index(embedding);

@time subspace(embedding, sample(embedding.vocab, 1_000));
@time subspace(embedding, sample(embedding.vocab, 10_000));
@time subspace(embedding2, sample(embedding.vocab, 1_000));
@time subspace(embedding2, sample(embedding.vocab, 10_000));
@time subspace(embedding2, sample(embedding.vocab, 111_111));

get(embedding2.dict, "дело_NOUN", "Token not found!")

get(embedding2, "дело_NOUN")
get(embedding3, "прочий_ADJ")
typeof(get(embedding3, "прочий_ADJ"))

# Benchmark Dictionary Creation
@benchmark embedding2 = IndexedWordEmbedding(embedding)
@benchmark embedding3 = index(embedding)

# Benchmark Indexing
@benchmark get(embedding2, token) setup = (token = sample(embedding.vocab, 1)[1])
@benchmark get(embedding3, token) setup = (token = sample(embedding.vocab, 1)[1])

# Benchmark Subspaces
@benchmark subspace(embedding, tokens) setup = (tokens = sample(embedding.vocab, 1_000))
@benchmark subspace(embedding2, tokens) setup = (tokens = sample(embedding.vocab, 1_000))

@benchmark subspace(embedding, tokens) setup = (tokens = sample(embedding.vocab, 10_000))
@benchmark subspace(embedding2, tokens) setup = (tokens = sample(embedding.vocab, 10_000))

@benchmark subspace(embedding, tokens) setup = (tokens = sample(embedding.vocab, 1_000))
@benchmark subspace(embedding2, tokens) setup = (tokens = sample(embedding.vocab, 1_000))
@benchmark subspace(embedding2, tokens) setup = (tokens = sample(embedding.vocab, 10_000))
### VERY POOR! ###
### UPD 27.12: Ok ###
