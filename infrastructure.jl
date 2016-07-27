type SpinHalfChain{T}
    L :: Int64
    X :: Array{SparseMatrixCSC{Float64         ,Int64}, 1}
    Y :: Array{SparseMatrixCSC{Complex{Float64},Int64}, 1}
    Z :: Array{SparseMatrixCSC{Float64         ,Int64}, 1}
    P :: Array{SparseMatrixCSC{Float64         ,Int64}, 1}
    M :: Array{SparseMatrixCSC{Float64         ,Int64}, 1}
    
    H_fn :: Function #Function returning instantaneous Hamiltonian at given time
    
    #Note: these are caches, and require updating!
    H :: SparseMatrixCSC{T, Int64} #Instantaneous Hamiltonian 
    H_eigendecomp #Instantaneous Hamiltonian eigendecomposition
end

function pauli_matrices(L :: Int64)
    I = speye(2)
    sigx = sparse([0 1; 1 0])
    sigy = sparse([0 -im; im 0])
    sigz = sparse([1 0; 0 -1])
    sigp = sparse([0 2; 0 0])
    sigm = sparse([0 0; 2 0])
    
    X = [reduce(kron, (speye(2^(j-1)), sigx, speye(2^(L - j)))) for j in 1:L]
    Y = [reduce(kron, (speye(2^(j-1)), sigy, speye(2^(L - j)))) for j in 1:L]
    Z = [reduce(kron, (speye(2^(j-1)), sigz, speye(2^(L - j)))) for j in 1:L]
    
    P = [reduce(kron, (speye(2^(j-1)), sigp, speye(2^(L - j)))) for j in 1:L]
    M = [reduce(kron, (speye(2^(j-1)), sigm, speye(2^(L - j)))) for j in 1:L]
    
    X = convert(Array{SparseMatrixCSC{Float64,Int64}, 1}, X)
    Y = convert(Array{SparseMatrixCSC{Complex{Float64},Int64}, 1}, Y)
    Z = convert(Array{SparseMatrixCSC{Float64,Int64}, 1}, Z)
    P = convert(Array{SparseMatrixCSC{Float64,Int64}, 1}, P)
    M = convert(Array{SparseMatrixCSC{Float64,Int64}, 1}, M)
    return (X,Y,Z,P,M)
end

function SpinHalfChain(L)
    pauli = pauli_matrices(L)
    H_fn = x -> speye(2^L)
    H = speye(2^L)
    H_eigendecomp = eigfact(full(H))
    return SpinHalfChain(L, pauli..., H_fn, H, H_eigendecomp)
end

function update!{T}(S :: SpinHalfChain{T}, H :: SparseMatrixCSC{T, Int64})
    S.H = H
    S.H_eigendecomp = eigfact(full(S.H))
end

function update!(S :: SpinHalfChain, t :: Float64)
    S.H = S.H_fn(t)
    S.H_eigendecomp = eigfact(full(S.H))
end
