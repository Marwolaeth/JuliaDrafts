using BenchmarkTools, BenchmarkPlots, StatsPlots
using LinearAlgebra

# DENSE ----

## Define and Check ----

a = Int16[1, 2, 3, 4]
b = Int16[2, 1, 2, 3]
x = [1.1, 2.2, 3.3, 4.4]
y = [2.1, 1.2, 2.3, 3.4]

norm(a) # 13!
√dot(a, a)
norm(b)

function cosine(x::Vector{<:AbstractFloat}, y::Vector{<:AbstractFloat})::Float64
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

function cosine(x::Vector{<:Integer}, y::Vector{<:Integer})::Float64
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

function cosine_basic(x, y)
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

function cosine(x::Vector{<:Real}, y::Vector{<:Real})::Float64
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

cosine(a, b)
cosine(a, a)
cosine(b, b)
cosine(a, b[1:3]) # DimensionMismatch Error
cosine(x, y)
cosine(x, x)
cosine(y, y)
cosine(y, x[1:3]) # DimensionMismatch Error
cosine(a, x)
cosine(b, y)
cosine(a, y)
cosine(x, b)
cosine(x, .-x)

## Benchmark ----
α = rand(Float64, 1000)
β = rand(Float64, 1000)
cosine(α, β)

@benchmark cosine(α, β)
@benchmark @noinline cosine_basic(α, β)

cosine_bench_setup = @benchmarkset "Cosine Benchmark" begin
    @case "Strong Typing" cosine($(Ref(α))[], $(Ref(β))[])
    @case "Weak Typing" cosine_basic($(Ref(α))[], $(Ref(β))[])
end
cosine_bench = run(cosine_bench_setup, samples=10000, verbose=true)
median(cosine_bench)
mean(cosine_bench)
maximum(cosine_bench)
minimum(cosine_bench)

# fieldnames(BenchmarkGroup)
# typeof(cosine_bench)
# methodswith(BenchmarkGroup)
# keys(cosine_bench)
# keys(cosine_bench["Cosine Benchmark"])
plot(cosine_bench["Cosine Benchmark"], logscale=true)
savefig("cosine-benchmark.png")
# plot(cosine_bench["cosine_bench"], logscale=false)

summary(cosine_bench["Cosine Benchmark"]["Strong Typing"])

# SPARSE ----
using SparseArrays

S = sprand(10, 26, 1 / 5)

typeof(S)
eltype(S)
typeof(S[1, :])
dot(S[1, :], S[3, :])
norm(S[1, :])

function cosine(x::SparseVector{<:AbstractFloat}, y::SparseVector{<:AbstractFloat})::Float64
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end
cosine(S[1, :], S[3, :])


function dropnan(x::SparseVector{<:AbstractFloat})::SparseVector{<:AbstractFloat}
    x[isnan.(x)] .= 0.0
    dropzeros(x)
end

function dropnan(X::SparseMatrixCSC{<:AbstractFloat})::SparseMatrixCSC{<:AbstractFloat}
    X[isnan.(X)] .= 0.0
    dropzeros(X)
end

function cosine(X::SparseMatrixCSC{<:AbstractFloat})::SparseMatrixCSC{Float64}
    n::Int32 = size(X, 1)
    ## Norms
    ℓ::SparseVector{Float64} = sparse([norm(X[i, :]) for i ∈ 1:n])
    dropnan((X * X') ./ (ℓ * ℓ'))
end

S = sprand(3, 5, 3 / 5)
# 111!
@time cosine(S)

cosine(S)

X = S
cosine(X)
X[2, :] = spzeros(5)
cosine(X)

C = (X * X') ./ (norms * norms')
C[isnan.(C)] .= spzeros(1)
dropnan(C)

@benchmark dropnan(C)

X = sprand(20, 200, 1 / 10)

function cosine2(X::SparseMatrixCSC{<:AbstractFloat})::SparseMatrixCSC{Float64}
    n::Int32 = size(X, 1)
    ## Norms
    ℓ::SparseVector{Float64} = sparse([norm(X[i, :]) for i ∈ 1:n])
    (X * X') ./ (ℓ * ℓ')
end

@benchmark cosine(X) setup = (X = sprand(20, 200, 1 / 10))
@benchmark cosine2(X) setup = (X = sprand(20, 200, 1 / 10))
# Only slight performance ban

function cosine3(X::SparseMatrixCSC{<:AbstractFloat})::SparseMatrixCSC{Float64}
    n::Int32 = size(X, 1)
    Xt = X'
    ## Norms
    ℓ::SparseVector{Float64} = sparse([norm(Xt[:, i]) for i ∈ 1:n])
    dropnan((X * Xt) ./ (ℓ * ℓ'))
end

@benchmark cosine(X) setup = (X = sprand(20, 200, 1 / 10))
@benchmark cosine3(X) setup = (X = sprand(20, 200, 1 / 10))
# Somehow rowwise norm calculation is better

@benchmark x[Ref(x .== 0.0).x] .= 1.0 setup = (x = Float64.([1, 0, 0, 1, 0, 1]))
@benchmark x[x.==0.0] .= 1.0 setup = (x = Float64.([1, 0, 0, 1, 0, 1]))

@benchmark [norm(Ref(S[i, :])) for i ∈ 1:5]
@benchmark [norm(S[i, :]) for i ∈ 1:5]

function safezeros!(x::Vector{Float64})
    @inbounds @fastmath x[x.==0.0] .= 1.0
end

x = [norm(S[i, :]) for i ∈ 1:size(S, 1)]
x[[1, 2]] .= 0.0
safezeros!(x)

function cosinepush!(
    C::SparseMatrixCSC{Float64,Int64},
    X::SparseMatrixCSC{Float64,Int64},
    ℓ::Vector{Float64}
)
    @inbounds @fastmath C .= (X * X') ./ (ℓ * ℓ')
end

function cosine4(X::SparseMatrixCSC{Float64,Int64})::SparseMatrixCSC{Float64,Int64}
    n = size(X, 1)
    # Preallocate C
    C = spzeros(Float64, n, n)
    # Preallocate norms
    ℓ::Vector{Float64} = Vector{Float64}(undef, n)
    ## Norms
    @inbounds @fastmath ℓ .= [norm(X[i, :]) for i ∈ 1:n]
    safezeros!(ℓ)
    @inbounds @fastmath cosinepush!(C, X, ℓ)
    return C
end
cosine4(S)

@time cosine4(S)
@time cosine(S)
cosine4(S) .== cosine(S)
norm(cosine4(S) - cosine(S))

@benchmark cosine(X) setup = (X = sprand(20, 200, 1 / 10))
@benchmark cosine4(X) setup = (X = sprand(20, 200, 1 / 10))

@benchmark cosine(X) setup = (X = sprand(111, 1111, 1 / 10))
@benchmark cosine4(X) setup = (X = sprand(111, 1111, 1 / 10))

S = sprand(1111, 111111, 1 / 5)
C = cosine(S)
C = cosine4(S)

@benchmark cosine(S) setup = (S = sprand(1111, 111111, 1 / 5)) seconds = 180
@benchmark cosine4(S) setup = (S = sprand(1111, 111111, 1 / 5)) seconds = 180

S = sprand(111, 1111, 1 / 5)
cosine4(S)
# compilation
@profview cosine4(S)
# pure runtime
@profview cosine4(S)

median(C)
mean(C)

using Plots, Statistics

std(C)
histogram(C.nzval[C.nzval.<0.999])

# SPECIAL PACKAGES
using LSHFunctions, Distances
cossim(X[1, :], X[3, :])
cosine_dist(X[1, :], X[3, :])

@benchmark cosine(X[1, :], X[3, :]) setup = (X = sprand(20, 200, 1 / 10))
@benchmark cossim(X[1, :], X[3, :]) setup = (X = sprand(20, 200, 1 / 10))
@benchmark cosine_dist(X[1, :], X[3, :]) setup = (X = sprand(20, 200, 1 / 10))

@benchmark cosine(X) setup = (X = sprand(50, 2000, 1 / 10))
@benchmark pairwise(CosineDist(), X', X') setup = (X = sprand(50, 2000, 1 / 10))
# Our functions performs better!

S = sprand(1111, 111111, 1 / 5)
@benchmark pairwise(CosineDist(), S', S')
C = cosine(S)
@benchmark cosine(S) setup = (S = sprand(1111, 111111, 1 / 5)) seconds = 180

# Tensors
using GPUArrays

# rowwise
function cosine(x::SparseVector{<:Integer}, y::SparseVector{<:Integer})::Float64
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

function cosine(x::SparseVector{<:Real}, y::SparseVector{<:Real})::Float64
    min(dot(x, y) / (norm(x) * norm(y)), 1.0)
end

S = sparse(
    [
        1.0 0.0 1.0 0.0;
        0.0 0.0 0.0 0.0;
        1.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0;
        0.0 1.0 1.0 0.0
    ]
)
n = size(Ref(S).x, 1)
C = sparse(1.0I, n, n)
ℓ = [norm(Ref(S[i, :]).x) for i ∈ 1:n]
ι = [i for (i, l) ∈ enumerate(Ref(ℓ).x) if l > 0]
eachindex(ι)
ℓ[ι]
isempty(ι)

for j ∈ ι[2:end], i ∈ ι[1:end-1]
    a = Ref(S[i, :]).x
    b = Ref(S[j, :]).x
    C[i, j] = C[j, i] = (a' * b) / (ℓ[i] * ℓ[j])
end
C
cosine4(S)
cosine([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

X = sprand(100, 2000, 1 / 10)
@benchmark a = Ref(X).x[i, :] setup = (i = 3)
@benchmark a = Ref(X[i, :]).x setup = (i = 3)

function cosinepush!(
    C::SparseMatrixCSC{Float64,Int64},
    X::SparseMatrixCSC{Float64,Int64},
    ℓ::Vector{Float64},
    ι::Vector{<:Integer}
)
    @inbounds @fastmath for j ∈ ι[2:end], i ∈ ι[1:end-1]
        a = Ref(X[j, :]).x
        b = Ref(X[j, :]).x
        C[i, j] = C[j, i] = (a' * b) / (ℓ[i] * ℓ[j])
    end
end

cosinepush!(C, S, ℓ, ι)
C

function cosine5(X::SparseMatrixCSC{Float64,Int64})::SparseMatrixCSC{Float64,Int64}
    n = size(Ref(X).x, 1)
    # Preallocate C
    C = sparse(1.0I, n, n)
    # Check if X is a zero matrix
    (length(X.nzval) == 0) && (return C)
    ## Norms
    @inbounds @fastmath ℓ::Vector{Float64} = [norm(Ref(X[i, :]).x) for i ∈ 1:n]
    ## Non-zero row indices
    @inbounds @fastmath ι = [i for (i, l) ∈ enumerate(Ref(ℓ).x) if l > 0]

    cosinepush!(C, X, ℓ, ι)
    return C
end

cosine5(S)

@benchmark cosine4(X) setup = (X = sprand(111, 1111, 1 / 10))
@benchmark cosine5(X) setup = (X = sprand(111, 1111, 1 / 10))

function cosine6(X::SparseMatrixCSC{Float64,Int64})::SparseMatrixCSC{Float64,Int64}
    n = size(X, 1)
    # Hard copy the transpose of X to process column-wise
    Xt = copy(X')
    # Preallocate C
    C = spzeros(Float64, n, n)
    # Preallocate norms
    ℓ::Vector{Float64} = Vector{Float64}(undef, n)
    ## Norms
    @inbounds @fastmath ℓ .= [norm(Xt[:, i]) for i ∈ 1:n]
    safezeros!(ℓ)
    @inbounds @fastmath cosinepush!(C, X, ℓ)
    return C
end

@benchmark cosine4(X) setup = (X = sprand(1_111, 11_111, 1 / 10))
@benchmark cosine6(X) setup = (X = sprand(1_111, 11_111, 1 / 10))

## Intime or After?
using SparseArrays
using BenchmarkTools
using LinearAlgebra
using StatsPlots
using HypothesisTests
include("cosine.jl")

S = sparse(
    [
        1.0 0.0 1.0 0.0;
        0.0 0.0 0.0 0.0;
        1.0 0.0 0.0 0.0
    ]
)

function ifzeros(X::SparseMatrixCSC)::Vector{Float64}
    n = size(X, 1)
    # Hard copy the transpose of X to process column-wise
    Xᵀ = copy(X')
    # Preallocate norms
    ℓ::Vector{Float64} = Vector{Float64}(undef, n)
    ## Norms
    @inbounds @fastmath [(l = norm(view(Xᵀ, :, i))) ≡ 0.0 ? 1.0 : l for i ∈ 1:n]
end

function safezeros(X::SparseMatrixCSC)::Vector{Float64}
    n = size(X, 1)
    # Hard copy the transpose of X to process column-wise
    Xᵀ = copy(X')
    # Preallocate norms
    ℓ::Vector{Float64} = Vector{Float64}(undef, n)
    ## Norms
    @inbounds @fastmath ℓ .= [norm(view(Xᵀ, :, i)) for i ∈ 1:n]
    Cosine.safezeros!(ℓ)
    return ℓ
end

ifzeros(S)
safezeros(S)

bench_intime = @benchmarkable ifzeros(S) seconds = 360 samples = 30 setup = (S = sprand(111_111, 2_222, 0.05))
# tune!(bench_local)

bench_after = @benchmarkable safezeros(S) seconds = 360 samples = 30 setup = (S = sprand(111_111, 2_222, 0.05))
# tune!(bench_pro)

res_bench_intime = run(bench_intime)
res_bench_after = run(bench_after)

times = [res_bench_intime.times; res_bench_after.times]
method = [
    repeat(["Inside"], length(res_bench_intime.times));
    repeat(["After"], length(res_bench_after.times))
]
boxplot(
    method,
    times,
    label=false
)

the_test = ExactMannWhitneyUTest(res_bench_intime.times, res_bench_after.times)

function cosine7(X::SparseMatrixCSC{Float64,Int64})::SparseMatrixCSC{Float64,Int64}
    n = size(X, 1)
    # Hard copy the transpose of X to process column-wise
    Xᵀ = copy(X')
    # Preallocate C
    C = spzeros(Float64, n, n)
    # Preallocate norms
    ℓ::Vector{Float64} = Vector{Float64}(undef, n)
    ## Norms
    @inbounds @fastmath ℓ .= [(l = norm(view(Xᵀ, :, i))) ≡ 0.0 ? 1.0 : l for i ∈ 1:n]
    @inbounds @fastmath Cosine.cosinepush!(C, X, ℓ)
    return C
end

cosine7(S)
Cosine.cosine(S)

bench_intime = @benchmarkable cosine7(S) seconds = 360 samples = 60 setup = (S = sprand(11_111, 2_222, 0.05))
# tune!(bench_local)

bench_after = @benchmarkable Cosine.cosine(S) seconds = 360 samples = 60 setup = (S = sprand(11_111, 2_222, 0.05))
# tune!(bench_pro)

# GOTO :)))

## Two Matrices ----
using SparseArrays
using Distances
include("cosine.jl")

S = sparse(
    [
        1.0 0.0 1.0 0.0;
        0.0 0.0 0.0 0.0;
        1.0 0.0 0.0 0.0
    ]
)

D = sparse(
    [
        1.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0;
        1.0 0.0 0.0 0.0
    ]
)

Cosine.cosine(S, D)
# Cosine.cosine(S, copy(D'))

## No intermediates ----
using LinearAlgebra
using SparseArrays
using Distances
using BenchmarkTools
using StatsPlots
using HypothesisTests
include("cosine.jl")

function cosine9(X::SparseMatrixCSC{Float64,Int64})::SparseMatrixCSC{Float64,Int64}
    n = size(X, 1)
    # Hard copy the transpose of X to process column-wise
    Xᵀ = copy(X')
    # Preallocate C
    C = spzeros(Float64, n, n)
    # Preallocate norms
    ℓ::Vector{Float64} = Vector{Float64}(undef, n)
    ## Norms
    @inbounds @fastmath ℓ .= [norm(view(Xᵀ, :, i)) for i ∈ 1:n]
    ℓ[ℓ.≡ 0.0] .= 1.0
    @inbounds @fastmath Cosine.cosinepush!(C, X, ℓ)
    return C
end

S = sprand(11_111, 222_222, 0.05)
S = sprand(1_111, 1_111_111, 1 / 50)
S = sprand(11_111, 1_111_111, 1 / 50)
Cosine.cosine(S)
cosine9(S)

@time Cosine.cosine(S)
@time cosine9(S)

function cosine10(X::SparseMatrixCSC{Float64,Int64})::SparseMatrixCSC{Float64,Int64}
    n = size(X, 1)
    # Hard copy the transpose of X to process column-wise
    Xᵀ = copy(X')
    # Preallocate C
    C = spzeros(Float64, n, n)
    # Preallocate norms
    ℓ::Vector{Float64} = Vector{Float64}(undef, n)
    ## Norms
    @inbounds @fastmath ℓ .= [norm(view(Xᵀ, :, i)) for i ∈ 1:n]
    @inbounds @fastmath ℓ[ℓ.≡ 0.0] .= 1.0
    @inbounds @fastmath C .= (X * X') ./ (ℓ * ℓ')
    return C
end

Cosine.cosine(S)
cosine10(S)

@time Cosine.cosine(S)
@time cosine10(S)

## PAIRS
using LinearAlgebra
using SparseArrays
include("cosine.jl")

S = sprand(666, 2_222, 0.05)
D = sprand(666, 2_222, 0.02)

Cosine.cosine(S => D)

@time Cosine.cosine(S => D)
@time Cosine.cosine(S, D)