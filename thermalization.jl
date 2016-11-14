#This way of using multiple dispatch is brittle and not naturally extensible.
#Better would be to define an abstract "coupling_scheme" type and a number of subtypes,
#but that would be ridiculous over-engineering, seems to me.

function coupling_factor(S :: AbstractSpinHalfChain, op :: Symbol,)
    if Symbol == :trivial
        L = sys.L
        return ones(2^L, 2^L)
    end
end

function coupling_factor(S :: AbstractSpinHalfChain, op :: SparseMatrixCSC,)
    V = S.H_eigendecomp[:vectors] #eigenvectors of Hamiltonian
    couple = abs(V'*op*V).^2
    couple = couple .* (couple .> 1e-14)
    return couple
end

function coupling_factor(S :: AbstractSpinHalfChain, scheme :: Tuple{Symbol, Int64})
    scheme, site = scheme
    assert(:proj_spec_site == scheme)
    V = S.H_eigendecomp[:vectors]#Hamiltonian eigenvectors
    X = V'*S.X[site]*V
    Y = V'*S.Y[site]*V
    Z = V'*S.Z[site]*V
    return sqrt(abs(X).^2 + abs(Y).^2 + abs(Z).^2)
end

#function coupling_factor(H_eigendecomp, scheme :: Symbol)
#    assert(:proj_onesitelocal = scheme)
#    error("Not implemented")
#end

#Note that for performance reasons
#    γ[η,α] is rate of η-->α
# (so later I can do γ[α,:]) but
#    Γ[η,α] is rate of α-->η.
function construct_γs(γoverall    :: Float64,
    β           :: Float64,
    sys         :: AbstractSpinHalfChain,
    coupling,
    bath_w      :: Float64)
    
    Es = sys.H_eigendecomp[:values]
    
    sys_bw = (maximum(Es) - minimum(Es))
    max_expt = 64
    Vs = sys.H_eigendecomp[:vectors]
    N = length(Es)
    γ = zeros(Float64, (N,N))
    for α in 1:N
        for η in 1:N
            if α != η && abs(Es[α] - Es[η]) < bath_w
                if β*abs(Es[α] - Es[η]) >= 40*log(2)
                    if Es[α] < Es[η]
                        γ[α, η] = γoverall * 1.0
                    else
                        γ[α, η] = 0 
                    end
                else
                    Zαη = exp(-β*(Es[α] - Es[η])/2) + exp(β*(Es[α] - Es[η])/2)
                    γ[α,η] = γoverall * exp(-β*(Es[α] - Es[η])/2)/Zαη
                end
            end
        end
    end

    return coupling_factor(sys, coupling) .* γ
end


#Assumes ρ is in the same basis in which H has eiegendecomposition H_eigdecomp
function thermalization_mats(γ :: Array{Float64, 2}, H_eigdecomp)
    E = H_eigdecomp[:values]
    N = size(E,1)
    spγ = sparse(γ)
    z = zeros(N,N)
    ω = zeros(N,N)
    for α = 1:N
        for η = 1:N
            if α != η
                z[α,η] = 0.5*(sum(spγ[:,α]) + sum(spγ[:,η]))
                ω[α,η] = E[α] - E[η]
            end
        end
    end
    
    Γ_diagcorr = zeros(N)
    for α = 1:N
        Γ_diagcorr[α] = sum(γ[:,α])
    end
    
    Γ = γ - diagm(Γ_diagcorr)
    
    return (Γ, z, ω)
end

function decayrates(sys, Γ, z, op :: SparseMatrixCSC )
    N = 2^sys.L
    assert(size(Γ) == (N,N))
    assert(size(z) == (N,N))
    U = sys.H_eigendecomp[:vectors]'
    op = U'*op*U
    op_diag = diag(op)
    op_offdiag = op - diagm(op_diag)
    
    rates = reshape(z, 2^(2*L))
    op_weights = reshape(op_offdiag, 2^(2*L))
    
    dΓ, vΓ = eig(Γ)
    vcat(op_weights, vΓ\op_diag)
    vcat(rates, dΓ)
    assert(length(op_weights)==length(rates))
    return (rates, op_weights)
end

#Note: assumes ρ in energy eigenbasis*
function thermalize(sys, Γ, z, ω, ρ, t)
    ρdiag = diag(ρ)
    ρoffdiag = ρ - diagm(ρdiag)
    
    ρoffdiag = ρoffdiag .* exp(-t*(z + im*ω))
    ρdiag    = expm(t*Γ)*ρdiag

    return ρoffdiag + diagm(ρdiag)
end

chain(f, lst) = [f(x2, x1) for (x1, x2) in zip(lst[1:end-1], lst[2:end])]
function decaytime(t, op_expect :: Array{Float64, 1}, gibbs_op_expect :: Float64)
    dts = chain(-, t)
    normalization = abs(op_expect[1] - gibbs_op_expect)
    diffs = chain((x,y) -> (x + y)/2, abs(op_expect - gibbs_op_expect))
    return sum(dts .* diffs)/normalization
end
