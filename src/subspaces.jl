using StatsBase
using BenchmarkTools, Plots, StatsPlots, BenchmarkPlots

cd(raw"ContentAnalysis")
include("readvec.jl")

path = "vectores/ruscorpora_upos_skipgram_300_5_2018.vec"
@time embed = ReadVec.read_vec(path)
embed.ndims
embed.ntokens

embedding = ReadVec.IndexedWordEmbedding(embed)
embedding.ntokens
embedding.ndims

# Draft
tokens = sample(embedding.vocab, 13)
inds = [findfirst(@inbounds embedding.vocab .≡ t) for t ∈ tokens]

@benchmark [findfirst(@inbounds embedding.vocab .≡ t) for t ∈ tokens] setup = (tokens = sample(embedding.vocab, 1_000))
@benchmark argmax.([@inbounds embedding.vocab .≡ t for t ∈ tokens]) setup = (tokens = sample(embedding.vocab, 1_000))
@benchmark [findall(@inbounds embedding.vocab .≡ t) for t ∈ tokens] setup = (tokens = sample(embedding.vocab, 1_000))
# ind = embedding.vocab .≡ tokens[1]
# sum(ind)
# inds = argmax.([embedding.vocab .≡ t for t ∈ tokens])
# embedding.vocab[inds]
# subspace = @view embedding.embeddings[inds, :]

# all(ReadVec.get(embedding, tokens[1]) .≡ subspace[1, :])
# all(ReadVec.get(embedding, tokens[2]) .≡ subspace[2, :])

# Check
tokens = sample(embedding.vocab, 13)
@time subemb1 = ReadVec.subspace(embed, tokens)
subemb1.ntokens
subemb1.ndims
subemb1.vocab
subemb1.embeddings
ReadVec.get(embedding, tokens[1])

@time subemb2 = ReadVec.subspace(embedding, tokens)
subemb2.ntokens
subemb2.ndims
subemb2.vocab
subemb2.embeddings

all(subemb2.embeddings .≡ subemb1.embeddings)
