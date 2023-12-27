module Cosine
export cosine

using LinearAlgebra

## DENSE ----

### VECTORS ----

function cosine(x::Vector{Float32}, y::Vector{Float32})::Float32
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

function cosine(
    x::SubArray{Float32,1,Matrix{Float32},Tuple{Int64,Base.Slice{Base.OneTo{Int64}}},true},
    y::SubArray{Float32,1,Matrix{Float32},Tuple{Int64,Base.Slice{Base.OneTo{Int64}}},true}
)::Float32
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

function cosine(x::Vector{Float64}, y::Vector{Float64})::Float64
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

function cosine(
    x::SubArray{Float64,1,Matrix{Float64},Tuple{Int64,Base.Slice{Base.OneTo{Int64}}},true},
    y::SubArray{Float64,1,Matrix{Float64},Tuple{Int64,Base.Slice{Base.OneTo{Int64}}},true}
)::Float64
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

function cosine(x::Vector{<:AbstractFloat}, y::Vector{<:AbstractFloat})::Float64
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

function cosine(
    x::SubArray{<:AbstractFloat,1,Matrix{<:AbstractFloat},Tuple{Int64,Base.Slice{Base.OneTo{Int64}}},true},
    y::SubArray{<:AbstractFloat,1,Matrix{<:AbstractFloat},Tuple{Int64,Base.Slice{Base.OneTo{Int64}}},true}
)::Float64
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

function cosine(x::Vector{Int64}, y::Vector{Int64})::Float64
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

function cosine(x::Vector{<:Integer}, y::Vector{<:Integer})::Float64
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

function cosine(
    x::SubArray{<:Integer,1,Matrix{<:Integer},Tuple{Int64,Base.Slice{Base.OneTo{Int64}}},true},
    y::SubArray{<:Integer,1,Matrix{<:Integer},Tuple{Int64,Base.Slice{Base.OneTo{Int64}}},true}
)::Float64
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

function cosine(x::Vector{<:Real}, y::Vector{<:Real})::Float64
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

function cosine(
    x::SubArray{<:Real,1,Matrix{<:Real},Tuple{Int64,Base.Slice{Base.OneTo{Int64}}},true},
    y::SubArray{<:Real,1,Matrix{<:Real},Tuple{Int64,Base.Slice{Base.OneTo{Int64}}},true}
)::Float64
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

### MATRICES ----
function cosine(X::Matrix{Float32})::Matrix{Float32}
    n = size(X, 1)
    # Hard copy the transpose of X to process column-wise
    Xᵀ = copy(X')
    # Preallocate C
    C = zeros(Float32, n, n)
    # Preallocate norms
    ℓ::Vector{Float32} = Vector{Float32}(undef, n)
    ## Norms
    @inbounds @fastmath ℓ .= [norm(view(Xᵀ, :, i)) for i ∈ 1:n]
    @inbounds @fastmath C .= (X * X') ./ (ℓ * ℓ')
    return C
end

function cosine(X::Matrix{Float64})::Matrix{Float64}
    n = size(X, 1)
    # Hard copy the transpose of X to process column-wise
    Xᵀ = copy(X')
    # Preallocate C
    C = zeros(Float64, n, n)
    # Preallocate norms
    ℓ::Vector{Float64} = Vector{Float64}(undef, n)
    ## Norms
    @inbounds @fastmath ℓ .= [norm(view(Xᵀ, :, i)) for i ∈ 1:n]
    @inbounds @fastmath C .= (X * X') ./ (ℓ * ℓ')
    return C
end

function cosine(X::Matrix{Int32})::Matrix{Float32}
    n = size(X, 1)
    # Hard copy the transpose of X to process column-wise
    Xᵀ = copy(X')
    # Preallocate C
    C = zeros(Float32, n, n)
    # Preallocate norms
    ℓ::Vector{Float32} = Vector{Float32}(undef, n)
    ## Norms
    @inbounds @fastmath ℓ .= [norm(view(Xᵀ, :, i)) for i ∈ 1:n]
    @inbounds @fastmath C .= (X * X') ./ (ℓ * ℓ')
    return C
end

function cosine(X::Matrix{Int64})::Matrix{Float64}
    n = size(X, 1)
    # Hard copy the transpose of X to process column-wise
    Xᵀ = copy(X')
    # Preallocate C
    C = zeros(Float64, n, n)
    # Preallocate norms
    ℓ::Vector{Float64} = Vector{Float64}(undef, n)
    ## Norms
    @inbounds @fastmath ℓ .= [norm(view(Xᵀ, :, i)) for i ∈ 1:n]
    @inbounds @fastmath C .= (X * X') ./ (ℓ * ℓ')
    return C
end

function cosine(X::Matrix{<:AbstractFloat})::Matrix{Float32}
    n = size(X, 1)
    # Hard copy the transpose of X to process column-wise
    Xᵀ = copy(X')
    # Preallocate C
    C = zeros(Float32, n, n)
    # Preallocate norms
    ℓ::Vector{Float32} = Vector{Float32}(undef, n)
    ## Norms
    @inbounds @fastmath ℓ .= [norm(view(Xᵀ, :, i)) for i ∈ 1:n]
    @inbounds @fastmath C .= (X * X') ./ (ℓ * ℓ')
    return C
end

function cosine(X::Matrix{<:Integer})::Matrix{Float32}
    n = size(X, 1)
    # Hard copy the transpose of X to process column-wise
    Xᵀ = copy(X')
    # Preallocate C
    C = zeros(Float32, n, n)
    # Preallocate norms
    ℓ::Vector{Float32} = Vector{Float32}(undef, n)
    ## Norms
    @inbounds @fastmath ℓ .= [norm(view(Xᵀ, :, i)) for i ∈ 1:n]
    @inbounds @fastmath C .= (X * X') ./ (ℓ * ℓ')
    return C
end

## Package Expansion: Only use if SparseArrays are installed
using SparseArrays

## SPARSE ----

### VECTORS ----

function cosine(x::SparseVector{Float32}, y::SparseVector{Float32})::Float32
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

function cosine(x::SparseVector{Float64}, y::SparseVector{Float64})::Float64
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

function cosine(x::SparseVector{<:AbstractFloat}, y::SparseVector{<:AbstractFloat})::Float64
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

function cosine(x::SparseVector{Int64}, y::SparseVector{Int64})::Float64
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

function cosine(x::SparseVector{<:Integer}, y::SparseVector{<:Integer})::Float32
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

function cosine(x::SparseVector{<:Real}, y::SparseVector{<:Real})::Float32
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

### MATRICES ----

function cosine(X::SparseMatrixCSC{Float64,Int64})::SparseMatrixCSC{Float64,Int64}
    n = size(X, 1)
    # Hard copy the transpose of X to process column-wise
    Xᵀ = copy(X')
    # Preallocate C
    C = spzeros(Float64, n, n)
    # Preallocate norms
    ℓ::Vector{Float64} = Vector{Float64}(undef, n)
    ## Norms
    @inbounds @fastmath ℓ .= [norm(view(Xᵀ, :, i)) for i ∈ 1:n]
    @inbounds @fastmath ℓ[ℓ.≡0.0] .= 1.0
    @inbounds @fastmath C .= (X * X') ./ (ℓ * ℓ')
    return C
end

function cosine(X::SparseMatrixCSC{Int64,Int64})::SparseMatrixCSC{Float64}
    n = size(X, 1)
    # Hard copy the transpose of X to process column-wise
    Xᵀ = copy(X')
    # Preallocate C
    C = spzeros(Float64, n, n)
    # Preallocate norms
    ℓ::Vector{Float64} = Vector{Float64}(undef, n)
    ## Norms
    @inbounds @fastmath ℓ .= [norm(view(Xᵀ, :, i)) for i ∈ 1:n]
    @inbounds @fastmath ℓ[ℓ.≡0.0] .= 1.0
    @inbounds @fastmath C .= (X * X') ./ (ℓ * ℓ')
    return C
end

function cosine(X::SparseMatrixCSC{<:Integer,<:Integer})::SparseMatrixCSC{Float32}
    n = size(X, 1)
    # Hard copy the transpose of X to process column-wise
    Xᵀ = copy(X')
    # Preallocate C
    C = spzeros(Float32, n, n)
    # Preallocate norms
    ℓ::Vector{Float32} = Vector{Float32}(undef, n)
    ## Norms
    @inbounds @fastmath ℓ .= [norm(view(Xᵀ, :, i)) for i ∈ 1:n]
    @inbounds @fastmath ℓ[ℓ.≡0.0] .= 1.0
    @inbounds @fastmath C .= (X * X') ./ (ℓ * ℓ')
    return C
end

function cosine(X::SparseMatrixCSC{Float32,Int64})::SparseMatrixCSC{Float32,Int64}
    n = size(X, 1)
    # Hard copy the transpose of X to process column-wise
    Xᵀ = copy(X')
    # Preallocate C
    C = spzeros(Float32, n, n)
    # Preallocate norms
    ℓ::Vector{Float32} = Vector{Float32}(undef, n)
    ## Norms
    @inbounds @fastmath ℓ .= [(l = norm(view(Xᵀ, :, i))) ≡ 0.0 ? 1.0 : l for i ∈ 1:n]
    @inbounds @fastmath C .= (X * X') ./ (ℓ * ℓ')
    return C
end

### MATRIX PAIRS ----

function cosine(
    S::SparseMatrixCSC{Float32,T},
    D::SparseMatrixCSC{Float32,T}
)::SparseMatrixCSC{Float32,T} where {T}
    # Early check for dimensions to match
    size(S) ≡ size(D) || error(
        "Dimension Mismatch:\n", "The matrices should be of the same dimensionality"
    )
    # Row-vector lengths
    n = size(S, 1)
    # Preallocate the resulting matrx
    C = spzeros(Float32, n, n)
    # Preallocate the norms products matrix
    N = spzeros(Float32, n, n)
    # Preallocate the norms array
    L::Array{Float32,2} = Matrix{Float32}(undef, 2, n)
    #= Pack the hard copies of the transposes
       so that they could be accepted column-wise inside a loop =#
    Tr = (copy(S'), copy(D'))

    # The norms
    @inbounds @fastmath L .= [
        (l = norm(view(X, :, i))) ≡ 0.0 ? 1.0 : l for X ∈ Tr, i ∈ 1:n
    ]
    # The norms products matrix
    @inbounds @fastmath N .= view(L, 1, :) * L[2, :]'
    # The pairwise similarity matrix
    @inbounds @fastmath C .= (S * D') ./ N
    return C
end

function cosine(
    S::SparseMatrixCSC{Float64,Int64},
    D::SparseMatrixCSC{Float64,Int64}
)::SparseMatrixCSC{Float64,Int64}
    # Early check for dimensions to match
    size(S) ≡ size(D) || error(
        "Dimension Mismatch:\n", "The matrices should be of the same dimensionality"
    )
    # Row-vector lengths
    n = size(S, 1)
    # Preallocate the resulting matrx
    C = spzeros(Float64, n, n)
    # Preallocate the norms products matrix
    N = spzeros(Float64, n, n)
    # Preallocate the norms array
    L::Array{Float64,2} = Matrix{Float64}(undef, 2, n)
    #= Pack the hard copies of the transposes
       so that they could be accepted column-wise inside a loop =#
    T = (copy(S'), copy(D'))

    # The norms
    @inbounds @fastmath L .= [
        (l = norm(view(X, :, i))) ≡ 0.0 ? 1.0 : l for X ∈ T, i ∈ 1:n
    ]
    # The norms products matrix
    @inbounds @fastmath N .= view(L, 1, :) * L[2, :]'
    # The pairwise similarity matrix
    @inbounds @fastmath C .= (S * D') ./ N
    return C
end

function cosine(
    S::SparseMatrixCSC{<:Integer,<:Integer},
    D::SparseMatrixCSC{<:Integer,<:Integer}
)::SparseMatrixCSC{Float32,Int64}
    # Early check for dimensions to match
    size(S) ≡ size(D) || error(
        "Dimension Mismatch:\n", "The matrices should be of the same dimensionality"
    )
    # Row-vector lengths
    n = size(S, 1)
    # Preallocate the resulting matrx
    C = spzeros(Float32, n, n)
    # Preallocate the norms products matrix
    N = spzeros(Float32, n, n)
    # Preallocate the norms array
    L::Array{Float32,2} = Matrix{Float32}(undef, 2, n)
    #= Pack the hard copies of the transposes
       so that they could be accepted column-wise inside a loop =#
    T = (copy(S'), copy(D'))

    # The norms
    @inbounds @fastmath L .= [
        (l = norm(view(X, :, i))) ≡ 0.0 ? 1.0 : l for X ∈ T, i ∈ 1:n
    ]
    # The norms products matrix
    @inbounds @fastmath N .= view(L, 1, :) * L[2, :]'
    # The pairwise similarity matrix
    @inbounds @fastmath C .= (S * D') ./ N
    return C
end

function cosine(
    S::SparseMatrixCSC{<:Real,T},
    D::SparseMatrixCSC{<:Real,T}
)::SparseMatrixCSC{Float32,T} where {T}
    # Early check for dimensions to match
    size(S) ≡ size(D) || error(
        "Dimension Mismatch:\n", "The matrices should be of the same dimensionality"
    )
    # Row-vector lengths
    n = size(S, 1)
    # Preallocate the resulting matrx
    C = spzeros(Float32, n, n)
    # Preallocate the norms products matrix
    N = spzeros(Float32, n, n)
    # Preallocate the norms array
    L::Array{Float32,2} = Matrix{Float32}(undef, 2, n)
    #= Pack the hard copies of the transposes
       so that they could be accepted column-wise inside a loop =#
    Tr = (copy(S'), copy(D'))

    # The norms
    @inbounds @fastmath L .= [
        (l = norm(view(X, :, i))) ≡ 0.0 ? 1.0 : l for X ∈ Tr, i ∈ 1:n
    ]
    # The norms products matrix
    @inbounds @fastmath N .= view(L, 1, :) * L[2, :]'
    # The pairwise similarity matrix
    @inbounds @fastmath C .= (S * D') ./ N
    return C
end

function cosine(M::Pair{SparseMatrixCSC{Float64,Int64},SparseMatrixCSC{Float64,Int64}})::SparseMatrixCSC{Float64,Int64}
    S, D = M
    cosine(S, D)
end

function cosine(M::Pair{SparseMatrixCSC{Float32,T},SparseMatrixCSC{Float32,T}})::SparseMatrixCSC{Float32,T} where {T}
    S, D = M
    cosine(S, D)
end

function cosine(M::Pair{SparseMatrixCSC{<:Real,T},SparseMatrixCSC{<:Real,T}})::SparseMatrixCSC{Float32,T} where {T}
    S, D = M
    cosine(S, D)
end

function cosine(M::Pair{SparseMatrixCSC{<:Integer,<:Integer},SparseMatrixCSC{<:Integer,<:Integer}})::SparseMatrixCSC{Float32,Int64}
    S, D = M
    cosine(S, D)
end

end