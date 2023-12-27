# Binary Files
using JLD2
cd(raw"C:\Users\apavlyuchenko\Documents\Projects\Julia\ContentAnalysis")
include("readvec.jl")

path = "vectores/ruscorpora_upos_skipgram_300_5_2018.vec"
@time embed = ReadVec.read_vec(path)
embed.ndims
embed.ntokens
embed.embeddings
typeof(eachcol(embed.embeddings)[1])

# Write
embed = ReadVec.read_vec(path)
jldsave(
    "vectores/ruscorpora_upos_skipgram_300_5_2018.jld";
    embedding=embed
)

# Read
f = jldopen("vectores/ruscorpora_upos_skipgram_300_5_2018.jld", "r")
embed = f["embedding"]
embed.vocab
embed.embeddings

@time embed = jldopen("vectores/ruscorpora_upos_skipgram_300_5_2018.jld", "r")["embedding"];

embedding2 = ReadVec.IndexedWordEmbedding(embed);
embedding2
embedding2.dict
ReadVec.getsafe(embedding2, "поверх_ADV")
ReadVec.getsafe(embedding2, "нихуя")

tokens = ["поверх_ADV", "всевышние_ADJ", "куб_ADJ", "хуй_NOUN", "хуйня_NOUN", "ерунда_NOUN", "досконале_ADV"]

tokens = intersect(tokens, embedding.vocab)

# [ReadVec.getsafe(embedding2, t) for t ∈ tokens]

C = cat([ReadVec.getsafe(embedding2, t) for t ∈ tokens]...; dims=2)
Cosine.cosine(copy(C'))
ReadVec.get(embedding2, tokens[4])