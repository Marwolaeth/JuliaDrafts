using Distances
using BenchmarkTools, Plots, StatsPlots, BenchmarkPlots

# Set WD
cd(raw"ContentAnalysis")
path = "vectores/test_skipgram_1111_300.vec"
# readline(path)
include("readvec.jl")
include("cosine.jl")

# Warm-up, Check & Play
@time Cosine.cosine(
    ReadVec.get(embedding2, "дело_NOUN"),
    ReadVec.get(embedding3, "прочий_ADJ")
)

@time Cosine.cosine(
    ReadVec.get(embedding2, "дело_NOUN"),
    ReadVec.get(embedding3, "рассказывать_VERB")
)

@time Cosine.cosine(
    ReadVec.get(embedding2, "дело_NOUN"),
    ReadVec.get(embedding2, "делать_VERB")
)
# ReadVec.get(embedding2, "сделать_VERB")
@time Cosine.cosine(
    ReadVec.get(embedding2, "играть_VERB"),
    ReadVec.get(embedding2, "делать_VERB")
)

# Only with large vectore
@time Cosine.cosine(
    ReadVec.get(embedding2, "делать_VERB"),
    ReadVec.get(embedding2, "сделать_VERB")
)

@time Cosine.cosine(
    ReadVec.get(embedding2, "дело_NOUN"),
    ReadVec.get(embedding2, "время_NOUN")
)

@time Cosine.cosine(
    ReadVec.get(embedding2, "дело_NOUN"),
    ReadVec.get(embedding2, "игра_NOUN")
)

# Only with large vectore
@time Cosine.cosine(
    ReadVec.get(embedding2, "дело_NOUN"),
    ReadVec.get(embedding2, "занятие_NOUN")
)

@time Cosine.cosine(
    ReadVec.get(embedding2, "дело_NOUN"),
    ReadVec.get(embedding2, "работа_NOUN")
)

# Out-of-memory on large vectore
@time C = Cosine.cosine(embedding.embeddings)

using LinearAlgebra
argmax(C - I(1111))

embedding3.vocab[1060]
embedding3.vocab[1013]

@time Cosine.cosine(
    ReadVec.get(embedding3, "сентябрь_NOUN"),
    ReadVec.get(embedding2, "август_NOUN")
)

C[argmax(C - I(1111))]
