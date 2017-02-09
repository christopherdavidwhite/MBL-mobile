function Pdiabatic(δETH, δMBL, v, δ )
    σ0 = speye(2)
    σx = sparse([0 1; 1 0])
    σy = sparse([0 -im; im 0])
    σz = sparse([1 0; 0 -1])
    H_fn(α) = (1 - α)*δETH*σx + α*δMBL*σz
    U = linear_expm_evolution(2, H_fn, 0, 1, v, δ)
    return abs(([1,0]' * U * [1;-1])[1])^2/2 #/2 is normalization for the -x
end


function Pdiabatic_reverse(δETH, δMBL, v, δ )
    σ0 = speye(2)
    σx = sparse([0 1; 1 0])
    σy = sparse([0 -im; im 0])
    σz = sparse([1 0; 0 -1])
    H_fn(α) = α*δETH*σx + (1-α)*δMBL*σz
    U = linear_expm_evolution(2, H_fn, 0, 1, v, δ)
    return abs(([1,1]' * U * [0,1])[1])^2/2 #/2 is normalization for the +x
end

function timestep(sys :: AbstractSpinHalfChain,
                  t :: Float64,
                  δ :: Float64,
                  U :: Array{Complex{Float64}, 2})
    return expm(full(-im*sys.H_fn(t)*δ))* U
end

function timestep(bond_evals :: Array{Float64,1},
                  bond_evects :: Array{Complex{Float64},2},
                  field :: Array{Float64, 1},
                  h :: Float64,
                  δ :: Float64,
                  U :: Array{Complex{Float64}, 2})
    expm_bond = scale(bond_evects, exp(-im*δ*bond_evals)) * bond_evects'
    U = scale(expm_bond, exp(-im*h*δ*field)) * U
    U :: Array{Complex{Float64}, 2}
    return U
end

function forwardalpha_timeevolution(sys :: AbstractSpinHalfChain, T :: Float64, δ :: Float64)
    U = eye(Complex{Float64}, 2^sys.L)
    for t in 0:δ:(T-δ)
        U = timestep(sys, t/T, δ, U)
    end
    update!(sys, 1.0)
    return U
end

function backwardalpha_timeevolution(sys :: AbstractSpinHalfChain, T :: Float64, δ :: Float64)
    U = eye(Complex{Float64}, 2^sys.L)
    for t in T:δ:δ
        U = timestep(sys, t/T, δ, U)
    end
    update!(sys, 0.0)
    return U
end

function forwardalpha_timeevolution(sys :: RFHeis, T :: Float64, δ :: Float64)
    U = eye(Complex{Float64}, 2^sys.L)
    hfn = sys.h
    Qfn = sys.scale
    bond_evals = sys.bond_evals
    bond_evects = sys.bond_evects
    field = sys.field
    for t in 0:δ:(T-δ)
        h = (hfn(t/T) :: Float64)
        Q = (Qfn(h) :: Float64)
        U = timestep(bond_evals, bond_evects, field, h, δ*Q, U)
    end
    update!(sys, 1.0)
    return U
end

function backwardalpha_timeevolution(sys :: RFHeis, T :: Float64, δ :: Float64)
    U = eye(Complex{Float64}, 2^sys.L)
    hfn = sys.h
    Qfn = sys.scale
    bond_evals = sys.bond_evals
    bond_evects = sys.bond_evects
    field = sys.field
    for t in T:-δ:δ
        h = (hfn(t/T) :: Float64)
        Q = (Qfn(h) :: Float64)
        U = timestep(bond_evals, bond_evects, field, h, δ*Q, U)
    end
    update!(sys, 0.0)
    return U
end

#This particular use of multiple dispatch (T as string for
#less-than-straightforward, e.g. adiabatic, time evolution) might be
#too clever by half.

function forwardalpha_timeevolution(sys :: AbstractSpinHalfChain, T :: AbstractString, δ :: Float64)
    if "adiabatic" == T
        #Trick is to write unitaries that'll do what I want: move me
        #from diagonal in one basis to diagonal in the other.

        #assumes already updated to t = 0
        
        d0, V0 = sys.H_eigendecomp[:values], sys.H_eigendecomp[:vectors]
        p0 = sortperm(d0) #make sure eigenvalues are sorted
        V0 = V0[p0, p0]

        update!(sys, 1.0)
        
        d1, V1 = sys.H_eigendecomp[:values], sys.H_eigendecomp[:vectors]
        p1 = sortperm(d1)
        V1 = V1[p1, p1]
        
        U = V1 * V0'
        #Can see that this is what we want: maps ground state of H0 to
        #ground state of H1, first excited to first excited, etc.
    else
        error("Unsupported time evolution $T!")
    end
    return U
end

function backwardalpha_timeevolution(sys :: AbstractSpinHalfChain, T :: AbstractString, δ :: Float64)
    if "adiabatic" == T
        #Trick is to write unitaries that'll do what I want: move me
        #from diagonal in one basis to diagonal in the other.

        #Assumes already updated to α = 1
        d1, V1 = sys.H_eigendecomp[:values], sys.H_eigendecomp[:vectors]
        p1 = sortperm(d1)
        V1 = V1[p1, p1]

        
        update!(sys, 0.0)
        
        d0, V0 = sys.H_eigendecomp[:values], sys.H_eigendecomp[:vectors]
        p0 = sortperm(d0) #make sure eigenvalues are sorted
        V0 = V0[p0, p0]
        
        
        U = V0 * V1'
    else
        error("Unsupported time evolution $T!")
    end
    return U
end
