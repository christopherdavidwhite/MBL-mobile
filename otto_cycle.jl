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
    V :: Array{Complex{Float64},2}
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
                             vs ,
                             Δtths :: Array{Float64, 1},
                             βHs   :: Array{Float64, 1},
                             βCs   :: Array{Float64, 1},
    N_reals :: Int64,
    conserving = false,
    filling_fraction = 0.5, #only if conserving
    δs  :: Array{Float64, 1} = [1/32],
    print_per = 100,
    )
    @show δs
    # sys.X[1] + sys.Z[1]
    if conserving
        sys = ConservingRFHeis(L, filling_fraction)
    else
        sys = RFHeis(L)
    end
    
    coupling_op = coupling_op_fn(sys)
    
    Q = h -> sqrt(3(L - 1) + L*h.^2/3)
    @show wbs

    coupling = coupling_op
    newdatavect(T) = zeros(T, N_reals*length(h0s)*length(wbs)*length(δs)*length(βHs)*length(βCs)*length(Δtths)*length(vs))
    data = Dict(
                :WETHMBL => newdatavect(Float64),
                :WMBLETH => newdatavect(Float64),
                :WTOT    => newdatavect(Float64),
                :QMBL    => newdatavect(Float64),
                :QETH    => newdatavect(Float64),
                :η       => newdatavect(Float64),
                :r       => newdatavect(Int64),
                :neg_work => newdatavect(Float64),
                :h0      => newdatavect(Float64),
                :h1      => newdatavect(Float64),
                :βH      => newdatavect(Float64),
                :βC      => newdatavect(Float64),
                :Δtths   => newdatavect(Float64),
                :v       => newdatavect(Float64),
                :wb      => newdatavect(Float64),
    :std  => newdatavect(Float64),
    :δ    => newdatavect(Float64),
    :comptime => newdatavect(Float64),
    :L => newdatavect(Float64),
    :field1 => newdatavect(Float64),
    :field2 => newdatavect(Float64),
    )

    c = 0
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
                c += 1
                for k in keys(dct)
                    data[k][c] =  real(dct[k])
                end
                data[:h0][c]    =  h0 
                data[:βC][c]    =  βC 
                data[:βH][c]    =  βH 
                data[:Δtths][c] =  Δtth
                data[:h1][c]    =  h1
                data[:v][c]     =  v
                data[:wb][c]    =  wb 
                data[:r][c]     =  r 
                data[:std][c]   =  std(d)
                data[:δ][c]     =  δ 
                data[:L][c]     =  L 
                data[:comptime][c] =  comptime
            end
        end
        if 0 == r % print_per
            @show (L, r, data[:comptime][end])
        end
    end

    return DataFrame(data);
end


