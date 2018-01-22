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
                  J :: Float64,
                  δ :: Float64,
                  U :: Array{Complex{Float64}, 2},
                  Y :: Array{Complex{Float64}, 2}, #storage for multiplication results
                  trotter_order :: Number,)
    if trotter_order == 1
        Ac_mul_B!(Y, bond_evects,  U)
        scale!(exp.(-im*δ*J*bond_evals), Y)
        A_mul_B!(U, bond_evects, Y)
        scale!(exp.(-im*δ*h*field), U)
    elseif trotter_order == 2
        expfield_2 = exp.(-im*δ*h*field/2)
        scale!(expfield_2, U)
        Ac_mul_B!(Y, bond_evects,  U)
        scale!(exp.(-im*δ*J*bond_evals), Y)
        A_mul_B!(U, bond_evects, Y)
        scale!(expfield_2, U)
    elseif trotter_order == Inf
        N = size(Y,1)
        #don't play the allocation game here: will be overwhelmed by expm1
        H = bond_evects * diagm(J*bond_evals) * bond_evects' + diagm(h*field)
        U = expm(-im*δ*H)*U
    else
        error("unsupported trotter order $(trotter_order)")
    end
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

function forwardalpha_timeevolution(sys :: RFHeis,
        T :: Float64,
        δ :: Float64,
        trotter_order :: Number = Inf,
        integrator_order = 2,
    )
        
    N = size(sys.Z[1],1)
    U = eye(Complex{Float64}, N)
    Y = zeros(Complex{Float64}, N,N)
    hfn = sys.h
    Qfn = sys.scale
    bond_evals = sys.bond_evals
    bond_evects = sys.bond_evects
    field = sys.field
    for t in 0:δ:T-δ
        if integrator_order == 1
            Qinv = sys.scale(t/T)
            J = Qinv
            h = Qinv * sys.h(t/T)
        elseif integrator_order == 2
            Qinv1 = sys.scale(t/T)
            Qinv2 = sys.scale((t-δ)/T)
            J1 = Qinv1
            J2 = Qinv2

            h1 = Qinv1 * sys.h(t/T)
            h2 = Qinv2 * sys.h((t-δ)/T)
            
            J = 0.5*(J1 + J2)
            h = 0.5*(h1 + h2)
        elseif integrator_order == 4
            error("this is not actually a fourth-order integrator")
            Qinv1 = sys.scale(t/T)
            Qinv2 = sys.scale((t - δ/2)/T)
            Qinv3 = sys.scale((t - δ)/T)
            
            J1 = Qinv1
            J2 = Qinv2
            J3 = Qinv3
            
            h1 = Qinv1 * sys.h(t/T)
            h2 = Qinv2 * sys.h((t - δ/2)/T)
            h3 = Qinv3 * sys.h((t - δ)/T)
            
            h = 1/6 * (h1 + 4*h2 + h3)
            J = 1/6 * (J1 + 4*J2 + J3)
        else
            error("no supported integrator at order $integrator_order")
        end
        U = timestep(bond_evals, bond_evects, field, h, J, δ, U, Y, trotter_order)
    end
    update!(sys, 1.0)
    return U
end

function backwardalpha_timeevolution(sys :: RFHeis,
        T :: Float64,
        δ :: Float64,
        trotter_order :: Number = Inf,
        integrator_order = 2,
    )
        
    N = size(sys.Z[1],1)
    U = eye(Complex{Float64}, N)
    Y = zeros(Complex{Float64}, N,N)
    hfn = sys.h
    Qfn = sys.scale
    bond_evals = sys.bond_evals
    bond_evects = sys.bond_evects
    field = sys.field
    for t in T:-δ:δ
        if integrator_order == 1
            Qinv = sys.scale(t/T)
            J = Qinv
            h = Qinv * sys.h(t/T)
        elseif integrator_order == 2
            Qinv1 = sys.scale(t/T)
            Qinv2 = sys.scale((t-δ)/T)
            J1 = Qinv1
            J2 = Qinv2

            h1 = Qinv1 * sys.h(t/T)
            h2 = Qinv2 * sys.h((t-δ)/T)
            
            J = 0.5*(J1 + J2)
            h = 0.5*(h1 + h2)
        elseif integrator_order == 4
            error("this is not actually a fourth-order integrator")
            Qinv1 = sys.scale(t/T)
            Qinv2 = sys.scale((t - δ/2)/T)
            Qinv3 = sys.scale((t - δ)/T)
            
            J1 = Qinv1
            J2 = Qinv2
            J3 = Qinv3
            
            h1 = Qinv1 * sys.h(t/T)
            h2 = Qinv2 * sys.h((t - δ/2)/T)
            h3 = Qinv3 * sys.h((t - δ)/T)
            
            h = 1/6 * (h1 + 4*h2 + h3)
            J = 1/6 * (J1 + 4*J2 + J3)
        else
            error("no supported integrator at order $integrator_order")
        end
        U = timestep(bond_evals, bond_evects, field, h, J, δ, U, Y, trotter_order)
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
