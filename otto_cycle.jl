function ediag(sys, ρ)
    V = sys.H_eigendecomp[:vectors]
    return real(diag(V'*ρ*V))
end

function otto_efficiency(sys  :: AbstractSpinHalfChain,
                         coupling,
                         δ    :: Number,
                         wb   :: Number,
                         T,
                         Δtth :: Number,
                         βH   :: Number,
                         βC   :: Number,
                         verbose :: Bool =false,
                         )

    L = sys.L

    
    #Pull out the fields on the first two sites: for use in testing
    #hacky and unidiomatic
    if sys.L == 2 && typeof(sys) == RFHeis
        field_frontfactor = sys.scale(0.0) * sys.h(0) * 2.0^sys.L
        field1 = trace(sys.H_fn(0.0) * sys.Z[1])/field_frontfactor
        field2 = trace(sys.H_fn(0.0) * sys.Z[2])/field_frontfactor
    else
        field1 = 0.0
        field2 = 0.0
    end
    
    #SETUP (ETH THERMALIZATION)
    update!(sys, 0.0)
    ρinitial = gibbs(sys.H_fn(0.0), βH)
    
    if verbose
        update!(sys, 0.0)
        @show T
        @show T/δ
        println("beginning")
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
    #V :: Array{Complex{Float64},2}
    ρE = V' * ρ * V
    ρp = V * thermalize(sys, βC, Γ, z, ω, ρE, Δtth) * V'
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
    
    neg_work = false
    Wtot = WETHMBL + WMBLETH
    Q = max(real(QETH), real(QMBL))
    if real(Wtot) < -1e-14
        neg_work = true
        η = NaN
    elseif abs(Q) < 1e-10
        if verbose
            println("zero heat transfer (presumably no-op)")
        end
        η = NaN
    else
        η = (WETHMBL + WMBLETH)/Q
    end

    check(QETH + QMBL - (WETHMBL + WMBLETH ), 0, "total energy exchange")
    out = Dict(:WETHMBL  => WETHMBL,
               :WMBLETH  => WMBLETH,
               :WTOT     => Wtot,
               :QETH     => QETH,
               :QMBL     => QMBL,
               :η        => η,
               :neg_work => neg_work,
               :field1   => field1,
               :field2   => field2,
               )

    return out
end

function map_otto_efficiency(L    :: Int64,
                             h1   :: Float64,
                             h0s  :: Array{Float64,1},
                             coupling_op_fn :: Function,
                             wbs  :: Array{Float64,1},
                             Ts ,
                             Δtths :: Array{Float64, 1},
                             βHs   :: Array{Float64, 1},
                             βCs   :: Array{Float64, 1},
    N_reals :: Int64,
    jldfile, 
    conserving = false,
    δs  :: Array{Float64, 1} = [1/32],
    print_per = 100,
    s = 0,
    )
    @show δs

    # sys.X[1] + sys.Z[1]
    if conserving #half-filling
        sys = ConservingRFHeis(L, 0.5)
        Q = h -> sqrt(3*L - 2 + (L - 2)/(L - 1) +L.*h.^2/3)
    else
        sys = RFHeis(L)
        Q = h -> sqrt(3(L - 1) + L*h.^2/3)
    end

    @show typeof(Q)
    coupling_op = coupling_op_fn(sys)
    
    @show wbs

    coupling = coupling_op
    newdatavect(T) = zeros(T, N_reals*length(h0s)*length(wbs)*length(δs)*length(βHs)*length(βCs)*length(Δtths)*length(Ts))
    for q in ["WETHMBL", "WMBLETH", "WTOT", "QMBL", "QETH", "η", "neg_work", "h0", "h1", "βH", "βC", "Δtths", "T", "wb", "std", "δ", "comptime" , "L" , "field1" , "field2"]
        jldfile[q] = newdatavect(Float64)
    end
    for q in ["r"]
        jldfile[q] = newdatavect(Int64)
    end
    
    c = 0
    @show N_reals
    tic()
    @time for r in 1:N_reals
        for h0 in h0s
            # label the disorder by the next integer from the PRNG,
            # immediately before we go get the fields
            disorderlabel = rand(Int)
            rfheis!(sys, h0, h1, Q)
            d = sys.H_eigendecomp[:values]
            for wb in wbs, δ in δs, βH in βHs, βC in βCs, Δtth in Δtths, T in Ts
                if T == Inf
                    Tp = "adiabatic"
                else
                    Tp = T
                end
                dct = otto_efficiency(sys, coupling_op, δ, wb, Tp, Δtth, βH, βC, false)
                comptime = 0
                c += 1
                for k in keys(dct)
                    jldfile[string(k)][c] =  real(dct[k])
                end
                jldfile["h0"][c]    =  h0 
                jldfile["βC"][c]    =  βC 
                jldfile["βH"][c]    =  βH 
                jldfile["Δtths"][c] =  Δtth
                jldfile["h1"][c]    =  h1
                jldfile["T"][c]     =  T
                jldfile["wb"][c]    =  wb 
                jldfile["r"][c]     =  disorderlabel 
                jldfile["std"][c]   =  std(d)
                jldfile["δ"][c]     =  δ 
                jldfile["L"][c]     =  L 
                jldfile["comptime"][c] =  comptime
            end
        end
        if 0 == r % print_per
            @show (L, r, toq())
            tic()
        end
    end
end


