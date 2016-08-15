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
    return scale(expm_bond, exp(-im*h*δ*field)) * U
end

function forwardalpha_timeevolution(sys :: AbstractSpinHalfChain, T :: Float64, δ :: Float64)
    U = eye(Complex{Float64}, 2^sys.L)
    for t in 0:δ:(T-δ)
        U = timestep(sys, t/T, δ, U)
    end
    return U
end

function backwardalpha_timeevolution(sys :: AbstractSpinHalfChain, T :: Float64, δ :: Float64)
    U = eye(Complex{Float64}, 2^sys.L)
    for t in T:δ:δ
        U = timestep(sys, t/T, δ, U)
    end
    return U
end

function forwardalpha_timeevolution(sys :: RFHeis, T :: Float64, δ :: Float64)
    U = eye(Complex{Float64}, 2^sys.L)
    for t in 0:δ:(T-δ)
        U = timestep(sys.bond_evals, sys.bond_evects, sys.field, sys.h(t/T), δ*sys.scale(sys.h(t/T)), U)
    end
    return U
end

function backwardalpha_timeevolution(sys :: RFHeis, T :: Float64, δ :: Float64)
    U = eye(Complex{Float64}, 2^sys.L)
    for t in T:δ:δ
        U = timestep(sys.bond_evals, sys.bond_evects, sys.field, sys.h(t/T), δ*sys.scale(sys.h(t/T)), U)
    end
    return U
end
