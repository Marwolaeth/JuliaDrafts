using BenchmarkTools
using SparseArrays

# GENERAL ----
bench = BenchmarkGroup([:index, :loop])

nz_index!(x::Vector{Float64}) = @inbounds @fastmath x[x.≡0.0] .= 1.0
function nz_loop!(x::Vector{Float64})
    @inbounds @fastmath for i ∈ 1:length(x)
        x[i] ≡ 0.0 && (x[i] = 1.0)
    end
end

x = sprandn(10, 0.1)
x = Float64.([x...])
# x = Float64.([2, 0, 2, 0, 2])
nz_index!(x)
x
x = Float64.([2, 0, 2, 0, 2])
nz_loop!(x)
x

bench[:index] = @benchmarkable nz_index!(x) samples = 100 setup = (x = Float64.([sprandn(1_000_000, 0.1)...]))
bench[:loop] = @benchmarkable nz_loop!(x) samples = 100 setup = (x = Float64.([sprandn(1_000_000, 0.1)...]))
bench

results = run(bench, verbose=true, seconds=20)
m_loop = median(results[:loop])
m_index = median(results[:index])
judge(m_index, m_loop)

# WITHIN FUNCTION ----
using LinearAlgebra
include("cosine.jl")

cosine = Cosine.cosine

function cosineB(X::SparseMatrixCSC{Float64,Int64})::SparseMatrixCSC{Float64,Int64}
    n = size(X, 1)
    # Hard copy the transpose of X to process column-wise
    Xᵀ = copy(X')
    # Preallocate C
    C = spzeros(Float64, n, n)
    # Preallocate norms
    ℓ::Vector{Float64} = Vector{Float64}(undef, n)
    ## Norms
    @inbounds @fastmath ℓ .= [norm(view(Xᵀ, :, i)) for i ∈ 1:n]
    @inbounds @fastmath nz_loop!(ℓ)
    @inbounds @fastmath C .= (X * X') ./ (ℓ * ℓ')
    return C
end

@btime cosine(sprand(1_111, 111_111, 0.1))
@btime cosineB(sprand(1_111, 111_111, 0.1))

bench = BenchmarkGroup()
bench[:current] = @benchmarkable cosine(S) samples = 30 setup = (S = sprand(1_111, 111_111, 0.1))
bench[:loop] = @benchmarkable cosineB(S) samples = 30 setup = (S = sprand(1_111, 111_111, 0.1))

results = run(bench, verbose=true, seconds=120)
m_new = median(results[:loop])
m_old = median(results[:current])
judge(m_old, m_new)

### No need for change