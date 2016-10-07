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
    if real(Wtot) < -1e-14
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

function map_otto_efficiency(L    :: Int64,
                             h1   :: Float64,
                             h0s  :: Array{Float64,1},
                             coupling_op_fn :: Function,
                             wbs  :: Array{Float64,1},
                             vs ,
                             Δtths :: Array{Float64, 1},
                             βHs   :: Array{Float64, 1},
                             βCs   :: Array{Float64, 1},
                             N_reals :: Int64,
    δs  :: Array{Float64, 1} = [1/32],
    print_per = 100
    )
    @show δs
    # sys.X[1] + sys.Z[1]
    sys = RFHeis(L)
    coupling_op = coupling_op_fn(sys)
    
    Q = h -> sqrt(3(L - 1) + L*h.^2/3)
    @show wbs

    coupling = coupling_op
    data = Dict(
                :WETHMBL => [],
                :WMBLETH => [],
                :WTOT    => [],
                :QMBL    => [],
                :QETH    => [],
                :η       => [],
                :r       => [],
                :neg_work => [],
                :h0      => [],
                :h1      => [],
                :βH      => [],
                :βC      => [],
                :Δtths   => [],
                :v       => [],
                :wb      => [],
    :std  => [],
    :δ    => [],
    :comptime => [],
    :L => [],
    )

    @show N_reals
    @time for r in 1:N_reals
        for h0 in h0s
            srand(r*10)
            rfheis!(sys, h0, h1, Q)
            update!(sys, 1.0)
            d = sys.H_eigendecomp[:values]
            for wb in wbs, δ in δs, βH in βHs, βC in βCs, Δtth in Δtths, v in vs
                if 0.0 == v
                    T = "adiabatic"
                else
                    T = abs((h0 - h1)/v)
                    N_steps = ceil(abs((h0 - h1)/(v*δ)))
                    δ = T/N_steps
                end
                tic()
                dct = otto_efficiency(sys, coupling_op, δ, wb, T, Δtth, βH, βC, false)
                comptime = toq()
                for k in keys(dct)
                    data[k] = push!(data[k], real(dct[k]))
                end
                data[:h0]    = push!(data[:h0]   , h0) 
                data[:βC]    = push!(data[:βC]   , βC) 
                data[:βH]    = push!(data[:βH]   , βH) 
                data[:Δtths] = push!(data[:Δtths], Δtths) 
                data[:h1]    = push!(data[:h1]   , h1) 
                data[:v]     = push!(data[:v]    , v)
                data[:wb]    = push!(data[:wb]   , wb) 
                data[:r]     = push!(data[:r]    , r) 
                data[:std]   = push!(data[:std]  , std(d)) 
                data[:δ]     = push!(data[:δ]    , δ) 
                data[:L]     = push!(data[:L]    , L) 
                data[:comptime] = push!(data[:comptime]    , comptime) 
            end
        end
        if 0 == r % print_per
            @show (L, r, data[:comptime][end])
        end
    end

    return DataFrame(data);
end


