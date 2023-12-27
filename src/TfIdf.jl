module TfIdf
export tf_idf

using LinearAlgebra, SparseArrays

A::SparseMatrixCSC{Int8,Int8} = sparse([1 0 1 0 1 0; 0 1 0 1 1 1; 0 1 1 1 1 0; 1 1 0 1 1 0])
N = size(A, 1)

tf = A .> 0 # Boolean frequencies
idf = log.(N .// sum(A, dims=1)) # or sum(tf, dims = 1)
tf_idf = tf .* idf

function tf_idf(X::SparseMatrixCSC{Float64,Int64})::SparseMatrixCSC{Float64,Int64}
    # tf_idf
end

end