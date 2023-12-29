__precompile__()

module ReadVec

export AbstractEmbedding
export WordEmbedding
export IndexedWordEmbedding
export read_vec, index, get, subspace


# using DelimitedFiles
using CSV

abstract type AbstractEmbedding end
struct WordEmbedding <: AbstractEmbedding
    embeddings::Matrix{Float32}  # Embeddings Matrix (transposed)
    vocab::Vector{String}        # Token Vocabulary
    ntokens::Int                 # Vocabulary Size
    ndims::Int                   # Embedding Dimensionality
end

struct IndexedWordEmbedding <: AbstractEmbedding
    embeddings::Matrix{Float32}  # Embeddings Matrix (transposed)
    vocab::Vector{String}        # Token Vocabulary
    dict::Dict{
        String,
        SubArray{Float32,1,Matrix{Float32},Tuple{Base.Slice{Base.OneTo{Int64}},Int64},true}
    }                            # Lookup Dictionary (Token ⟹ Embedding Vector)
    ntokens::Int                 # Vocabulary Size
    ndims::Int                   # Embedding Dimensionality

    # Constructed from unindexed embedding by adding a lookup dictionary
    function IndexedWordEmbedding(emb::WordEmbedding)
        d::Dict{
            String,
            SubArray{
                Float32,
                1,
                Matrix{Float32},
                Tuple{Base.Slice{Base.OneTo{Int64}},Int64},
                true
            }
        } = Dict((emb.vocab .=> eachcol(emb.embeddings))...)

        new(emb.embeddings, emb.vocab, d, emb.ntokens, emb.ndims)
    end
end

@inline function index(emb::WordEmbedding)::IndexedWordEmbedding
    IndexedWordEmbedding(emb)
end

"""
get(emb::IndexedWordEmbedding, query::String)

`get()` interface to indexed word embedding objects: returns embedding vector (Float32) for a given token `query`. Called with the embedding object rather than with the dictionary. Type-stable: returns a view of the embedding vector or throws an exception.
"""
@inline function get(
    emb::IndexedWordEmbedding,
    query::String
)::SubArray{
    Float32,
    1,
    Matrix{Float32},
    Tuple{Base.Slice{Base.OneTo{Int64}},Int64},
    true
}
    exception_string::String = "Token $query not found!"
    v = Base.get(emb.dict, query, exception_string)

    (typeof(v) ≡ String) && throw(ErrorException(exception_string))

    return v
end

"""
getsafe(emb::IndexedWordEmbedding, query::String)

For internal use only. This function is similar to `get()` but returns a zero vector if the `query` is not in the vocabulary.
"""
@inline function getsafe(
    emb::IndexedWordEmbedding,
    query::String
)::SubArray{
    Float32,
    1,
    Matrix{Float32},
    Tuple{Base.Slice{Base.OneTo{Int64}},Int64},
    true
}
    v = Base.get(emb.dict, query, "Token $query not found!")

    (typeof(v) ≡ String) && return eachcol(zeros(Float32, emb.ndims))[1]

    return v
end

"""
getsafe(emb::IndexedWordEmbedding, query::String)

For internal use only. This function is similar to `get(emb::IndexedWordEmbedding, query::String)`. However, it assumes that the `query` is definitely in the vocabulary and bypasses the check for speed and doesn't throw any meaningful exception.
"""
function get_unsafe(
    emb::IndexedWordEmbedding,
    query::String
)::SubArray{
    Float32,
    1,
    Matrix{Float32},
    Tuple{Base.Slice{Base.OneTo{Int64}},Int64},
    true
}
    Base.get(emb.dict, query, "Token $query not found!")
end

# function readvec(path::AbstractString)::WordEmbedding
#     vec = readdlm(path, ' ')
#     WordEmbedding(Float32.(vec[2:end, 2:end]), String.(vec[2:end, 1]))
# end

"""
read_vec(path::AbstractString)::WordEmbedding

Reads a local embedding vector from `path` and creates a `WordEmbedding` object using CSV.jl.
"""
@inline function read_vec(path::AbstractString)::WordEmbedding
    # Read dimensionality
    ntokens, ndims = parse.(Int, split(readline(path), ' '))

    # Pre-allocate
    emb = WordEmbedding(
        Array{Float32}(undef, ndims, ntokens),  # Embeddings Matrix (transposed)
        Array{String}(undef, ntokens),          # Token Vocabulary
        ntokens,                                # Vocabulary Size
        ndims                                   # Embedding Dimensionality
    )

    # Read CSV & Write Vectors column-wise
    @inbounds emb.embeddings .= CSV.Tables.matrix(
        CSV.File(
            path;
            skipto=2,
            delim=' ',
            header=false,
            types=Float32,
            select=2:(ndims+1)
        )
    )'

    # Note: is it Ok to read the file twice? How can we avoid it?
    @inbounds emb.vocab .= CSV.Tables.getcolumn(
        CSV.File(
            path;
            skipto=2,
            delim=' ',
            header=false,
            types=String,
            select=[1]
        ),
        1
    )[:, 1]
    return emb
end

# function readvec2(path::AbstractString)::WordEmbedding
#     # Read dimensionality
#     ntokens, ndims = parse.(Int, split(readline(path), ' '))

#     # Pre-allocate
#     emb = WordEmbedding(
#         Array{Float32}(undef, ntokens, ndims),
#         Array{String}(undef, ntokens)
#     )

#     # Read Lines & Write Vectors
#     open(path) do file
#         @inbounds for (i, ln) ∈ enumerate(eachline(file))
#             if i ≡ 1
#                 continue
#             end
#             ln = split(ln, ' ')
#             @inbounds emb.vocab[i-1] = ln[1]
#             @inbounds emb.embeddings[i-1, :] .= parse.(Float32, ln[2:end])
#         end
#     end

#     return emb
# end

"""
subspace(emb::AbstractEmbedding, tokens::Vector{String})::WordEmbedding

The `subspace()` function takes an existing embedding and a subset of its vocabulary as input and creates a new `WordEmbedding` object. The order of embedding vectors in the new embedding corresponds to the order of input `tokens`. If a token is not found in the source embedding vocabulary, a zero vector is returned for that token. 

It's worth noting that this method is relatively slow and doesn't assume the source embedding to be indexed. Therefore, it doesn't take advantage of a lookup dictionary. It's recommended to index an embedding before subsetting it for better performance.
"""
@inline function subspace(emb::AbstractEmbedding, tokens::Vector{String})::WordEmbedding
    # Clear out out-of-vocabulary tokens
    tokens::Vector{String} = intersect(tokens, emb.vocab)
    ntokens = length(tokens)

    # Pre-allocate
    sub = WordEmbedding(
        Array{Float32}(undef, emb.ndims, ntokens),
        tokens,
        ntokens,
        emb.ndims
    )

    # Fill
    # inds::Vector{Int} = Array{Int}(undef, ntokens)
    @inbounds @simd for i ∈ 1:ntokens
        sub.embeddings[:, i] .=
            @view emb.embeddings[:, findfirst(emb.vocab .≡ tokens[i])]
    end
    # @inbounds inds .= [findfirst(@inbounds emb.vocab .≡ t) for t ∈ tokens]
    # @inbounds sub.embeddings .= emb.embeddings[:, inds]

    return sub
end

"""
subspace(emb::IndexedWordEmbedding, tokens::Vector{String})::WordEmbedding

The `subspace()` function takes an existing indexed embedding and a subset of its vocabulary as input and creates a new `WordEmbedding` object. It takes two arguments: `emb`, which is the existing indexed embedding, and `tokens`, which is a vector of strings representing the subset of vocabulary.

The order of embedding vectors in the new embedding corresponds to the order of input `tokens`. If a token is not found in the source embedding vocabulary, a zero vector is returned.

It is recommended to `index()` an embedding before subsetting it, as this method assumes that the source embedding is indexed and can use its lookup dictionary. This makes it relatively fast.

Note that the result is not an indexed embedding, so if the user wants it indexed, it needs to be done manually.
"""
@inline function subspace(emb::IndexedWordEmbedding, tokens::Vector{String})::WordEmbedding
    # Clear out out-of-vocabulary tokens
    # tokens::Vector{String} = intersect(tokens, emb.vocab)
    ntokens = length(tokens)

    # Pre-allocate
    sub = WordEmbedding(
        Array{Float32}(undef, emb.ndims, ntokens),
        tokens,
        ntokens,
        emb.ndims
    )

    # Fill
    # @inbounds sub.embeddings .= cat([getsafe(emb, t) for t ∈ tokens]...; dims=2)

    @inbounds @simd for i ∈ 1:ntokens
        sub.embeddings[:, i] .= getsafe(emb, tokens[i])
    end

    return sub
end

end