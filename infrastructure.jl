import Base.convert

tol = 1e-10

function check(val, target, name)
    err = val - target
    fail = maximum(abs(err)) > tol
    if fail
        println((name, val, target, err))
    end
    assert(!fail)
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
    H_eigendecomp :: Base.LinAlg.Eigen                     # Cache of eig'decomp. of Ham. at "current" α
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

abstract RFHeis <: AbstractSpinHalfChain
type NonconservingRFHeis{T} <: RFHeis
    L :: Int64                                             # System length
    X :: Array{SparseMatrixCSC{Float64         ,Int64}, 1} # List of onsite Pauli X matrices
    Y :: Array{SparseMatrixCSC{Complex{Float64},Int64}, 1} # -------------------- Y --------
    Z :: Array{SparseMatrixCSC{Float64         ,Int64}, 1} # -------------------- Z --------
    P :: Array{SparseMatrixCSC{Float64         ,Int64}, 1} # List of onsite raising  operators x + iy
    M :: Array{SparseMatrixCSC{Float64         ,Int64}, 1} # List of onsite lowering operators x - iy
    PM :: Array{SparseMatrixCSC{Float64        ,Int64}, 1} 

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
    field_mat :: Array{T,2}                                    # "field" term: part diagonal in comp. basis
    
    #Note: these are caches, and require updating!
    H :: Array{T, 2}                         # Cache of Hamiltonian at "current" α
    H_eigendecomp :: Base.LinAlg.Eigen                     # Cache of eig'decomp. of Ham. at "current" α
end

type ConservingRFHeis{T} <: RFHeis
    L :: Int64                                             # System length
    Z :: Array{SparseMatrixCSC{Float64         ,Int64}, 1} # -------------------- Z --------
    PM :: Array{SparseMatrixCSC{Float64        ,Int64}, 1} 

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
    field_mat :: Array{T,2}                                    # "field" term: part diagonal in comp. basis
    
    #Note: these are caches, and require updating!
    H :: Array{T, 2}                         # Cache of Hamiltonian at "current" α
    H_eigendecomp :: Base.LinAlg.Eigen                     # Cache of eig'decomp. of Ham. at "current" α
end

function invariant_checks(::RFHeis)
    parent_invariants = invariant_checks(AbstractSpinHalfChain)
    specific = [
                sys -> check(sys.H_fn(0), sys.scale(0) * ( sys.bond + sys.h(0) * spdiagm(sys.field) )),
                sys -> check(sys.bond, sys.bond_evects * sys.bond_evals, sys.bond_evects'),
                ]
    return [parent_invariants; specific]
end


#can't figure out how to make "convert" work
#take an RFHeis and write the equivalent SpinHalfChain
function despecialize(sys :: NonconservingRFHeis{Float64})
    return SpinHalfChain(sys.L,sys.X,sys.Y,sys.Z,sys.P,sys.M,sys.H_fn,sys.H,sys.H_eigendecomp)
end

# Compute (sparse) Pauli matrices and ladder operators for a chain of length L
function pauli_matrices(L :: Int64)
    sigx = sparse([0 1; 1 0])
    sigy = sparse([0 -im; im 0])
    sigz = sparse([1 0; 0 -1])
    sigp = sparse([0 1; 0 0])
    sigm = sparse([0 0; 1 0])
    
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

#Z, PM in sector with filling fraction f
function conserving_pauli_matrices(L :: Int64, f :: Float64)
    n = round(Int, L*f) #number of particles
    
    # We can imagine visiting the L choose n combinations of locations
    # (that is, the L choose n different configurations with exactly n
    # particles) in lexicographical order; this order defines a
    # mapping
    #
    #     state |---> order visited ~= basis vector.
    #
    # We need a way to compute this map: I've taken a configuration
    # and hopped a particle: what basis vector does the new state
    # correspond to?
    #
    # This function 'rank' does that. It is due to Lehmer; cf Knuth v4
    # fasc. 3 pp 5-6.
    #
    #For example:
    #
    # L = 6
    # n = 3
    # for c in combinations(1:L, 2)
    #     l = zeros(Int64, L)
    #     l[c] = 1
    #     l = reverse(l)
    #     @show l, rank(l)
    # end
    
    function rank(st :: Array{Int64})
        ct = reverse(find(st) - 1)
        t  = length(ct):-1:1
        return sum(map(binomial, ct, t)) + 1
    end

    #These may be slow. (It shouldn't matter, because I'll be re-using
    #the Paulis for every disorder realization and parameter.)
    #
    #  1. Chief problem: I probably don't insert elements in an order
    #  that's nice for a SparseMatrixCSC. This can probably be fixed
    #  by changing the order in which I walk through the basis states:
    #  want increasing rank (row or column? Don't know.) Might also
    #  want to separate out function
    #
    #  2. Might also want to pull inner loop into own function. This
    #  is lower priority.
    
    function spZ(j :: Int64, n :: Int64, L :: Int64)
        assert(1 <= j <= L)
        assert(1 <= n < L)
        N = binomial(L,n)
        Z = spzeros(N,N)
        for c in combinations(1:L, n)
            l = zeros(Int64, L)
            l[c] = 1
            b = rank(l)
            Z[b,b] = 2*l[j] - 1
        end
        return Z
    end

    function spPM(jP :: Int64, jM :: Int64, n :: Int64, L :: Int64)
        assert(1 <= jP <= L)
        assert(1 <= jM <= L)
        assert(1 <= n < L)
        N = binomial(L,n)
        PM = spzeros(N,N)
        for c in combinations(1:L, n)
            l = zeros(Int64, L)
            l[c] = 1
            b = rank(l)
            if ((1 == l[jM]) & (0 == l[jP]))
                lp = copy(l)
                lp[jM] = 0
                lp[jP] = 1
                bp = rank(lp)
                PM[bp,b] = 1
            end
        end
        return PM
    end

    Z = [spZ(j, n, L) for j in 1:L]
    PM = [spPM(jP, jM, n, L) for (jP, jM) in zip(1:L, circshift(1:L, -1))]
    return Z, PM
end

#Construct an "empty" (identity-Hamiltonian) SpinHalfChain of length L
function SpinHalfChain(L)
    pauli = pauli_matrices(L)
    H_fn = x -> speye(2^L)
    H = speye(2^L)
    H_eigendecomp = eigfact(full(H))
    return SpinHalfChain(L, pauli..., H_fn, H, H_eigendecomp)
end

#Construct an "empty" (identity-Hamiltonian) RFHeis of length L
function RFHeis(L)
    pauli = pauli_matrices(L)
    scale = α -> 1
    H_fn = x -> speye(2^L)
    H = eye(2^L)
    H_eigendecomp = eigfact(full(H))
    bond = zeros(2^L,2^L)
    bond_evals = zeros(2^L)
    bond_evects = eye(Complex{Float64}, 2^L)
    field = ones(2^L)
    field_mat = diagm(field)
    h = α -> 1
    PM = map(*, pauli[4], circshift(pauli[4], -1))
    return NonconservingRFHeis(L, pauli..., PM, H_fn, scale, h, bond, bond_evals, bond_evects, field, field_mat, H, H_eigendecomp)
end

#Construct an "empty" (identity-Hamiltonian) RFHeis of length L
function ConservingRFHeis(L :: Int64, f :: Float64)
    n = round(Int, L*f) #number of particles
    N = binomial(L,n)
    Z, PM = conserving_pauli_matrices(L, f)
    scale = α -> 1
    H_fn = x -> speye(N)
    H = eye(N)
    H_eigendecomp = eigfact(full(H))
    bond = zeros(N,N)
    bond_evals = zeros(N)
    bond_evects = eye(Complex{Float64}, N)
    field = ones(N)
    field_mat = diagm(field)
    h = α -> 1
    return ConservingRFHeis(L, Z, PM, H_fn, scale, h, bond, bond_evals, bond_evects, field, field_mat, H, H_eigendecomp)
end

function update!{T}(S :: AbstractSpinHalfChain, H :: SparseMatrixCSC{T, Int64})
    S.H = H
    S.H_eigendecomp = eigfact(full(S.H))
end

function update!(S :: RFHeis, α :: Float64)
    S.H = (S.bond + S.h(α) * (S.field_mat)) * S.scale(α)
    S.H_eigendecomp = eigfact(S.H)
end

function update!(S :: SpinHalfChain, t :: Float64)
    S.H = S.H_fn(t)
    S.H_eigendecomp = eigfact(full(S.H))
end

function gibbs(H :: SparseMatrixCSC, β :: Float64)
    ρ = expm(full(-β*H))
    ρ = ρ/trace(ρ)
    return ρ
end

#multiply Hamiltonian by 1/Q.
#Q is a function of h
function rfheis!(sys :: RFHeis, h0 :: Float64, h1 :: Float64, Q :: Function = h -> 1, bc = :open, bond_sgn = +1)
    L = sys.L
    (Z,PM) = (sys.Z, sys.PM)
    N = size(Z[1], 1)
    h = 2*rand(L) - 1

    bond = spzeros(Float64, N,N)
    for j in 1:(L - 1)
        bond += bond_sgn * (2 * (PM[j] + PM[j]') + Z[j] * Z[j+1])
    end
    if bc == :periodic
        bond += bond_sgn * (2 * (PM[L] + PM[L]') + Z[L] * Z[1])
    elseif bc != :open
        error("unrecognized bc ", bc)
    end

    field = spzeros(Float64, N,N)
    for j in 1:L
        field += Z[j]*h[j]
    end

    sys.bond_evals, sys.bond_evects = bond |> full|> eig
    sys.bond  = full(bond)
    sys.field = diag(field)
    sys.field_mat = full(field)
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
