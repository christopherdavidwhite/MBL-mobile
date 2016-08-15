function ediag(sys, ρ)
    V = sys.H_eigendecomp[:vectors]
    return real(diag(V'*ρ*V))
end

function otto_efficiency(sys  :: AbstractSpinHalfChain,
                         coupling,
                         δ    :: Number,
                         wb   :: Number,
                         T    :: Number,
                         Δtth :: Number,
                         βH   :: Number,
                         βC   :: Number,
                         verbose :: Bool =false,
                         )

    L = sys.L

    #SETUP (ETH THERMALIZATION)

    ρinitial = gibbs(sys.H_fn(0.0), βH)
    if verbose
        update!(sys, 0.0)
        @show T
        @show T/δ
        println("beginning")
        @show Es
        @show ediag(sys, ρinitial)
        @show round(ρinitial,5)
        @show sys.H
    end
    
    #ETH-->MBL
    U1 = forwardalpha_timeevolution(sys, T, δ)

    update!(sys, 1.0) #includes the diagonalization we'll need later
    
    ρ = U1*ρinitial*U1'
    WETHMBL = vecdot(ρinitial, sys.H_fn(0)) - vecdot(ρ, sys.H_fn(1))
    
    #MBL THERMALIZATION
    γ = construct_γs(1.0, βC, sys, coupling, wb)
    (Γ, z, ω) = thermalization_mats(γ, sys.H_eigendecomp)

    V = sys.H_eigendecomp[:vectors]
    V :: Array{Complex{Float64},2}
    ρE = V' * ρ * V
    ρp = V * thermalize(sys, Γ, z, ω, ρE, Δtth) * V'
    if verbose
        @show ediag(sys, ρ)
        @show real(diag(ρE))
        @show ediag(sys, ρp)
    end
    QMBL = vecdot(ρp, sys.H_fn(1)) - vecdot(ρ, sys.H_fn(1))

    #MBL-->ETH
    U3 = backwardalpha_timeevolution(sys, T, δ)
        
    ρpp = U3*ρp*U3'
    
    WMBLETH = vecdot(ρp, sys.H_fn(1)) - vecdot(ρpp, sys.H_fn(0))
    if verbose
        update!(sys, 0.0)
        Es = sys.H_eigendecomp[:values]
        V = sys.H_eigendecomp[:vectors]
        println("end")
        @show γ
        @show round(Es, 4)
        @show ediag(sys, ρpp)
        @show real(diag(V'*ρinitial*V))
        @show round(ρinitial,5)
        @show sys.H
        println()
    end
    QETH = vecdot(ρinitial, sys.H_fn(0)) - vecdot(ρpp, sys.H_fn(0))
    
    if verbose
        println("WETHMBL $WETHMBL")
        println("QMBL $QMBL")
        println("WMBLETH $WMBLETH")
        println("QETH $QETH")
    end
    
    Wtot = WETHMBL + WMBLETH
    if real(Wtot) < 0
        neg_work = true
        if verbose
            println("negative work: $δETH $δMBL $βH $βC $v $δ")
        end
        η = NaN
    else
        neg_work = false
        η = (WETHMBL + WMBLETH)/max(real(QETH), real(QMBL))
    end

    check(QETH + QMBL - (WETHMBL + WMBLETH ), 0, "total energy exchange")
    out = Dict(:WETHMBL  => WETHMBL,
               :WMBLETH  => WMBLETH,
               :WTOT     => Wtot,
               :QETH     => QETH,
               :QMBL     => QMBL,
               :η        => η,
               :neg_work => neg_work,
               )

    return out
end
