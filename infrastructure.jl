import Base.convert

tol = 1e-10

function check(val, target, name)
    diff = val - target
    fail = maximum(abs(diff)) > tol
    if fail
        println((name, val, target, diff))
    end
    assert(!fail)
end

function check(val :: Eigen, target :: Eigen, name)
    check(val
end

abstract AbstractSpinHalfChain

#Generic spin-1/2 chain. Fields:
type SpinHalfChain{T} <: AbstractSpinHalfChain
    L :: Int64                                               # System length
    X :: Array{SparseMatrixCSC{Float64         ,Int64}, 1}   # List of onsite Pauli X matrices
    Y :: Array{SparseMatrixCSC{Complex{Float64},Int64}, 1}   # -------------------- Y --------
    Z :: Array{SparseMatrixCSC{Float64         ,Int64}, 1}   # -------------------- Z --------
    P :: Array{SparseMatrixCSC{Float64         ,Int64}, 1}   # List of onsite raising  operators x + iy
    M :: Array{SparseMatrixCSC{Float64         ,Int64}, 1}   # List of onsite lowering operators x - iy
    
    
    # H_fn  : α ∈ [0,1] --> SparseMatrixCSC
    # scale : α ∈ [0,1] --> Float64
    # h     : α ∈ [0,1] --> Float64
    
    H_fn  :: Function                                      # Returns instantaneous Hamiltonian as fn of α
    
    #Note: these are caches, and require updating!
    H :: SparseMatrixCSC{T, Int64}                         # Cache of Hamiltonian at "current" α
    H_eigendecomp                                          # Cache of eig'decomp. of Ham. at "current" α
end

function invariant_checks(::AbstractSpinHalfChain)
    invs =  [sys -> check(sys.L, size(sys.X, 1), "sys.X length"),
             sys -> check(sys.L, size(sys.Y, 1), "sys.Y length"),
             sys -> check(sys.L, size(sys.Z, 1), "sys.Z length"),
             sys -> check(sys.L, size(sys.P, 1), "sys.P length"),
             sys -> check(sys.L, size(sys.M, 1), "sys.M length"),
                   
             sys -> check(sys.H, sys.H_eigendecomp[:vectors] * sys.H_eigendecomp[:values] * sys.H_eigendecomp[:vectors]'),
    sys -> check(size(sys.H), (2^sys.L, 2^sys.L)),
    sys -> check(size(sys.H_fn(0.0)), (2^sys.L, 2^sys.L)),
    ]
    return invs
end

type RFHeis{T} <: AbstractSpinHalfChain
    L :: Int64                                             # System length
    X :: Array{SparseMatrixCSC{Float64         ,Int64}, 1} # List of onsite Pauli X matrices
    Y :: Array{SparseMatrixCSC{Complex{Float64},Int64}, 1} # -------------------- Y --------
    Z :: Array{SparseMatrixCSC{Float64         ,Int64}, 1} # -------------------- Z --------
    P :: Array{SparseMatrixCSC{Float64         ,Int64}, 1} # List of onsite raising  operators x + iy
    M :: Array{SparseMatrixCSC{Float64         ,Int64}, 1} # List of onsite lowering operators x - iy


    # System's "real" Hamiltonian is
    #
    #    H(α) = scale(α) * ( bond + h(α) * hdiagm(field) )
    # 
    # H_fn returns exactly this: when you do α |> sys.H_fn |> eig,
    # you're diagonalizing the Hamiltonian at a tuning point α. We
    # keep track of the scale separately to make time evolution
    # easier.
    
    # H_fn  : α ∈ [0,1] --> SparseMatrixCSC
    # scale : α ∈ [0,1] --> Float64
    # h     : α ∈ [0,1] --> Float64
    
    H_fn  :: Function                                      # Returns instantaneous Hamiltonian as fn of α
    scale :: Function                                      # Returns overall scale as fn of α
    h     :: Function                                      # Returns magnitude of field: Hamiltonian =

    bond  :: Array{T,2}                                    # "Bond term": part off-diagonal in comp. basis
    bond_evals  :: Array{Float64,1}                        # Eigenvalues of bond term
    bond_evects :: Array{Complex{Float64},2}               # Eigenvectors of bond term
    field :: Array{T,1}                                    # "field" term: part diagonal in comp. basis
    
    #Note: these are caches, and require updating!
    H :: SparseMatrixCSC{T, Int64}                         # Cache of Hamiltonian at "current" α
    H_eigendecomp                                          # Cache of eig'decomp. of Ham. at "current" α
end

function invariant_checks(::RFHeis)
    parent = invariant_checks(AbstractSpinChain)
    specific = [
                sys -> check(sys.H_fn(0), sys.scale(0) * ( sys.bond + sys.h(0) * spdiagm(sys.field) )),
                sys -> check(bond, sys.bond_evects * sys.bond_evals, sys.bond_evects'),
                ]
    return [parent; specific]
end


#can't figure out how to make "convert" work
function despecialize(sys :: RFHeis{Float64})
    return SpinHalfChain(sys.L,sys.X,sys.Y,sys.Z,sys.P,sys.M,sys.H_fn,sys.H,sys.H_eigendecomp)
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

function RFHeis(L)
    pauli = pauli_matrices(L)
    scale = α -> 1
    H_fn = x -> speye(2^L)
    H = speye(2^L)
    H_eigendecomp = eigfact(full(H))
    bond = zeros(2^L,2^L)
    bond_evals = zeros(2^L)
    bond_evects = eye(Complex{Float64}, 2^L)
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
    sys.bond  = full(bond)
    sys.field = diag(field)
    sys.scale = α -> 1.0/Q(sys.h(α))
    sys.h     = α -> (h0 * (1 - α) + h1* α)
    sys.H_fn  = α -> (bond + sys.h(α) * (field)) * sys.scale(α)
    update!(sys, 0.0)
    return(bond, field)
end

function blank_data_hash(symbols :: Array{Symbol, 1})
    data = Dict()
    for s in symbols
        data[s] = []
    end
    return data
end
