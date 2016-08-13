tol = 1e-10

function check(val, target, name)
    diff = val - target
    fail = abs(diff) > tol
    if fail
        println((name, val, target, diff))
    end
    assert(!fail)
end

abstract AbstractSpinHalfChain

type SpinHalfChain{T} <: AbstractSpinHalfChain
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

type RFHeis{T} <: AbstractSpinHalfChain
    L :: Int64
    X :: Array{SparseMatrixCSC{Float64         ,Int64}, 1}
    Y :: Array{SparseMatrixCSC{Complex{Float64},Int64}, 1}
    Z :: Array{SparseMatrixCSC{Float64         ,Int64}, 1}
    P :: Array{SparseMatrixCSC{Float64         ,Int64}, 1}
    M :: Array{SparseMatrixCSC{Float64         ,Int64}, 1}
    
    H_fn  :: Function #Function returning instantaneous Hamiltonian at given time
    scale :: Function
    h     :: Function
    bond  :: Array{T,2}
    bond_evals  :: Array{Float64,1}
    bond_evects :: Array{T,2}
    field :: Array{T,1}
    
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
    return SpinHalfCHain(L, pauli..., H_fn, H, H_eigendecomp)
end

function RFHeis(L)
    pauli = pauli_matrices(L)
    scale = α -> 1
    H_fn = x -> speye(2^L)
    H = speye(2^L)
    H_eigendecomp = eigfact(full(H))
    bond = zeros(2^L,2^L)
    bond_evals = zeros(2^L)
    bond_evects = eye(2^L)
    field = zeros(2^L)
    h = α -> 1
    return RFHeis(L, pauli..., H_fn, scale, h, bond, bond_evals, bond_evects, field, H, H_eigendecomp)
end

function update!{T}(S :: AbstractSpinHalfChain, H :: SparseMatrixCSC{T, Int64})
    S.H = H
    S.H_eigendecomp = eigfact(full(S.H))
end

function update!(S :: AbstractSpinHalfChain, t :: Float64)
    S.H = S.H_fn(t)
    S.H_eigendecomp = eigfact(full(S.H))
end

function gibbs(H :: SparseMatrixCSC, β :: Float64)
    ρ = expm(full(-β*H))
    ρ = ρ/trace(ρ)
    return ρ
end

#multiply hamiltonian by 1/Q
function rfheis!(sys :: SpinHalfChain, hmax :: Float64, Q :: Float64)
    L = sys.L
    (M,P,Z) = (sys.M, sys.P, sys.Z)
    
    h = 2*rand(L) - 1

    bond = spzeros(Float64, 2^L, 2^L)
    for j in 1:(L - 1)
        bond += (P[j]*M[j+1] + M[j]P[j+1])/2 + Z[j] * Z[j+1]
    end
    #bond += (P[L]M[1] + M[L]P[1])/2 + Z[L] * Z[1]

    field = spzeros(Float64, 2^L, 2^L)
    for j in 1:L
        field += Z[j]*h[j]
    end

    update!(sys, bond + hmax*field)
    return(bond, field)
end

#multiply Hamiltonian by 1/Q.
#Q is a function of h
function rfheis!(sys :: RFHeis, h0 :: Float64, h1 :: Float64, Q :: Function = h -> 1)
    L = sys.L
    (M,P,Z) = (sys.M, sys.P, sys.Z)
    
    h = 2*rand(L) - 1

    bond = spzeros(Float64, 2^L, 2^L)
    for j in 1:(L - 1)
        bond += (P[j]*M[j+1] + M[j]P[j+1])/2 + Z[j] * Z[j+1]
    end
    #bond += (P[L]M[1] + M[L]P[1])/2 + Z[L] * Z[1]

    field = spzeros(Float64, 2^L, 2^L)
    for j in 1:L
        field += Z[j]*h[j]
    end

    bond_evals, bond_evects = bond |> full|> eig 
    sys.bond = bond
    sys.field = diag(field)
    sys.scale = α -> 1.0/Q(sys.h(α))
    sys.h     = α -> (h0 * (1 - α) + h1* α)
    sys.H_fn  = α -> (bond + sys.h(α) * (field)) * sys.scale(α)
    update!(sys, 0.0)
    return(bond, field)
end
