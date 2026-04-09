#Computer assisted proof of a branch of periodic solutions for the 1D Thomas model 
# The following code computes the solution and rigorously proves the results given in section 4.2 of
# "Proving the existence of localized patterns, periodic solutions, and branches of periodic solutions in the 1D Thomas model"  Dominic Blanco

# We provide the data for the approximate solution. 
# From this we can check if the proof of the solution is verified or not.

#####################################################################################################################################################################

# Needed packages
using RadiiPolynomial, LinearAlgebra, JLD2

function Fₚ!(Fₚ,U,ν,ν₁,ν₂,ν₃,ν₄,ν₅)
    U₁,U₂ = eachcomponent(U)
    gₚ = ν₄ - (ν₁*ν*U₁^2 - ν₁*U₁*U₂)*inv(1 + U₁ + ν₂*U₁^2)
    project!(component(Fₚ,1),Derivative(2)*U₁*ν - U₁ + gₚ)
    project!(component(Fₚ,2),Derivative(2)*U₂ - ν₃*U₂ + (ν₃*ν-1)*U₁ - ν₃*ν₅ + ν₄)
    return Fₚ
end

function DFₚ!(DFₚ,U,ν,ν₁,ν₂,ν₃)
    DFₚ .= 0
    U₁,U₂ = eachcomponent(U)
    Δ = project(Derivative(2),space(U₁),space(U₁),Float64)
    V₁ = -ν₁*(-U₂ + 2ν*U₁+ν*U₁^2 + ν₂*U₁^2*U₂)*inv(1 + U₁ + ν₂*U₁^2)^2
    V₂ = ν₁*U₁*inv(1+U₁+ν₂*U₁^2)
    𝕍ₚ₁ = project(Multiplication(V₁),space(U₁),space(U₁),Float64)
    𝕍ₚ₂ = project(Multiplication(V₂),space(U₁),space(U₁),Float64)
    project!(component(DFₚ,1,1),Δ*ν - I + 𝕍ₚ₁)
    project!(component(DFₚ,1,2), 𝕍ₚ₂)
    project!(component(DFₚ,2,1),(ν₃*ν-1)*I)
    project!(component(DFₚ,2,2),Δ-ν₃*I)
    return DFₚ
end

function _newton_s(u,jmax,ν,ν₁,ν₂,ν₃,ν₄,ν₅)
    T = similar(u)
    s = space(u)
    r = length(u)
    DT = LinearOperator(s,s,similar(coefficients(u),r,r))
    j = 0
    ϵ = 1
    nv = 1
    while (ϵ > 1e-13) & (j < jmax)
        T = Fₚ!(T,u,ν,ν₁,ν₂,ν₃,ν₄,ν₅)
        DT = DFₚ!(DT,u,ν,ν₁,ν₂,ν₃)
        u = u - DT\T
        @show ϵ = norm(T,Inf)
        nv = norm(u)
        if nv < 1e-5
            @show nv = norm(u)
            display("Newton may have converged to the 0 solution")
            return nv,j
            break
        end
        j += 1
    end
    return u,ϵ
end

function F_con!(F_con,X,Ū,U̇,ν,ν₁,ν₂,ν₃,ν₄)
    ν₅,U = eachcomponent(X) 
    component(F_con,1)[1] = sum((U - Ū).*U̇)
    project!(component(F_con,2), Fₚ!(similar(U),U,ν,ν₁,ν₂,ν₃,ν₄,ν₅[1]))
    return F_con
end

function F_con(W̄_cheb,Ẇ_cheb,gₚ_cheb,ν,ν₁,ν₂,ν₃,ν₄,N,N₀,d,NJ)
    pSpace0 = ParameterSpace() × CosFourier(N₀,π/d)^2
    pSpace = ParameterSpace() × CosFourier(N,π/d)^2
    Space = CosFourier(N,π/d)
    # Want F for NJ size 
    NJ_fft = nextpow(2,2NJ + 1)
    W̄_fft_NJ = cheb2grid(W̄_cheb,pSpace0,NJ_fft)
    Ẇ_fft_NJ = cheb2grid(Ẇ_cheb,pSpace0,NJ_fft)
    Ū_fft_NJ = component.(W̄_fft_NJ,2)
    Ū₁_fft_NJ = component.(component.(W̄_fft_NJ,2),1)
    Ū₂_fft_NJ = component.(component.(W̄_fft_NJ,2),2)
    ν₅_fft_NJ = getindex.(component.(W̄_fft_NJ,1),1)
    gₚ_fft_NJ = cheb2grid(gₚ_cheb,CosFourier(4N₀,π/d),NJ_fft)
    F_con1 = Vector{Sequence{CartesianProduct{Tuple{ParameterSpace, CartesianPower{CosFourier{Interval{Float64}}}}}, Vector{Interval{Float64}}}}(undef,NJ_fft)
    for j = 1:NJ_fft
        Fcon1 = project(Derivative(2)*Ū₁_fft_NJ[j]*ν - Ū₁_fft_NJ[j] + gₚ_fft_NJ[j],Space)
        Fcon2 = project(Derivative(2)*Ū₂_fft_NJ[j] - ν₃*Ū₂_fft_NJ[j] + (ν₃*ν-exact(1))*Ū₁_fft_NJ[j] - ν₃*ν₅_fft_NJ[j] + ν₄,Space)
        F_con1[j] = Sequence(pSpace, [interval(0) ; coefficients(Fcon1) ; coefficients(Fcon2)])
    end
    return F_con1
end

function L_con_tail(W̄_cheb,N,N₀,d,Nc,ν,ν₃)
    # Since we will be adding this to G_con_tail, we need 4Nc. 
    N4 = 4Nc 
    N4_fft = nextpow(2,2N4+1)
    Ū_fft = component.(cheb2grid(W̄_cheb,ParameterSpace() × CosFourier(N₀,π/d)^2,N4_fft),2)
    Ū₁_fft = component.(Ū_fft,1)
    Ū₂_fft = component.(Ū_fft,2)
    Ū₁_diff = Ū₁_fft .- project.(Ū₁_fft,CosFourier(N,π/d))
    Ū₂_diff = Ū₂_fft .- project.(Ū₂_fft,CosFourier(N,π/d))
    _tail_part1 = Derivative(2).*Ū₁_diff*ν .- Ū₁_diff
    _tail_part2 = Derivative(2).*Ū₂_diff .- ν₃*Ū₂_diff .+ (ν₃*ν-exact(1)).*Ū₁_diff 
    _tail_part1_cheb,_tail_part1_cheb0 = grid2cheb(_tail_part1,CosFourier(N₀,π/d),N4)
    _tail_part2_cheb,_tail_part2_cheb0 = grid2cheb(_tail_part2,CosFourier(N₀,π/d),N4)
    return _tail_part1_cheb0,_tail_part2_cheb0
end

function G_con_tail(gₚ_cheb,N,N₀,d,Nc)
    N4 = 4Nc
    N4_fft = nextpow(2,2N4+1)
    gₚ_fft_N4 = cheb2grid(gₚ_cheb,CosFourier(4N₀,π/d),N4_fft)
    _tail_part = gₚ_fft_N4 .- project.(gₚ_fft_N4,CosFourier(N,π/d))
    _tail_part_cheb,_tail_part_cheb0 = grid2cheb(_tail_part,CosFourier(4N₀,π/d),N4)
    return _tail_part_cheb0
end

function _Y0_con_tail(W̄_cheb,gₚ_cheb,N,N₀,d,Nc,ν,ν₃)
    N4 = 4Nc
    N4_fft = nextpow(2,2N4+1)
    gₚ_fft_N4 = cheb2grid(gₚ_cheb,CosFourier(4N₀,π/d),N4_fft)
    Ū_fft = component.(cheb2grid(W̄_cheb,ParameterSpace() × CosFourier(N₀,π/d)^2,N4_fft),2)
    Ū₁_fft = component.(Ū_fft,1)
    Ū₂_fft = component.(Ū_fft,2)
    Ū₁_diff = Ū₁_fft .- project.(Ū₁_fft,CosFourier(N,π/d))
    Ū₂_diff = Ū₂_fft .- project.(Ū₂_fft,CosFourier(N,π/d))
    _tail_part1 = Derivative(2).*Ū₁_diff*ν .- Ū₁_diff .+ gₚ_fft_N4 .- project.(gₚ_fft_N4,CosFourier(N,π/d))
    _tail_part2 = Derivative(2).*Ū₂_diff .- ν₃*Ū₂_diff .+ (ν₃*ν-exact(1)).*Ū₁_diff 
    _tail_part1_cheb,_tail_part1_cheb0 = grid2cheb(_tail_part1,CosFourier(4N₀,π/d),N4)
    _tail_part2_cheb,_tail_part2_cheb0 = grid2cheb(_tail_part2,CosFourier(N₀,π/d),N4)
    return _tail_part1_cheb0,_tail_part2_cheb0
end

function DF_con!(DF_con,X,U̇,ν,ν₁,ν₂,ν₃)
    ν₅,U = eachcomponent(X) 
    component(DF_con,1,2)[1,:] .= U̇
    project!(component(component(DF_con,2,1),2),project(-ν₃*one(component(U,1)),space(U)[1]))
    DF1 = LinearOperator(space(U),space(U), zeros(dimension(space(U)),dimension(space(U))))
    project!(component(DF_con,2,2), DFₚ!(DF1,U,ν,ν₁,ν₂,ν₃))
    return DF_con
end

function DF_con(X,U̇,ν,ν₁,ν₂,ν₃)
    DF_con1 = LinearOperator(space(X),space(X),zeros(dimension(space(X)),dimension(space(X))))
    ν₅,U = eachcomponent(X) 
    component(DF_con1,1,2)[1,:] .= U̇
    project!(component(component(DF_con1,2,1),2),project(-ν₃*one(component(U,1)),space(U)[1]))
    DF1 = LinearOperator(space(U),space(U), zeros(dimension(space(U)),dimension(space(U))))
    project!(component(DF_con1,2,2), DFₚ!(DF1,U,ν,ν₁,ν₂,ν₃))
    return DF_con1
end

function DFₚ_2N!(DFₚ_2N,U,ν,ν₁,ν₂,ν₃,Space2,Space1,V₁,V₂)
    DFₚ_2N .= interval.(0)
    U₁,U₂ = eachcomponent(U)
    Δ = project(Derivative(2),space(U₁),space(U₁),Interval{Float64})
    𝕍ₚ₁ = project(Multiplication(V₁),Space2,Space1,Interval{Float64})
    𝕍ₚ₂ = project(Multiplication(V₂),Space2,Space1,Interval{Float64})
    project!(component(DFₚ_2N,1,1),Δ*ν - UniformScaling(interval(1)) + 𝕍ₚ₁)
    project!(component(DFₚ_2N,1,2), 𝕍ₚ₂)
    project!(component(DFₚ_2N,2,1),(ν₃*ν-exact(1))*UniformScaling(interval(1)))
    project!(component(DFₚ_2N,2,2),Δ-ν₃*UniformScaling(interval(1)))
    return DFₚ_2N
end

function DF_con_2N(X,U̇,ν,ν₁,ν₂,ν₃,V₁,V₂)
    Space = space(X) 
    N = order(Space[2][1])
    f = frequency(Space[2][1])
    Space2 = ParameterSpace() × CosFourier(2N,f)^2
    DF_con2 = LinearOperator(Space2,Space,interval.(zeros(dimension(Space),dimension(Space2))))
    ν₅,U = eachcomponent(X) 
    component(DF_con2,1,2)[1,:] .= project(U̇,Space2[2])
    project!(component(component(DF_con2,2,1),2),project(-ν₃*one(component(U,1)),space(U)[1]))
    Space22 = Space2[2]
    Space12 = Space[2]
    DF1 = LinearOperator(Space22,Space12, interval.(zeros(dimension(Space12),dimension(Space22))))
    project!(component(DF_con2,2,2), DFₚ_2N!(DF1,U,ν,ν₁,ν₂,ν₃,Space22[1],Space12[1],V₁,V₂))
    return DF_con2
end

function TangentVector(X,Ẋ,ν,ν₁,ν₂,ν₃)
    ν₅,U = eachcomponent(X)
    U̇ = Sequence(space(U), [zeros(dimension(space(U)[1])) ; coefficients(project(-ν₃*one(component(U,1)),space(U)[1]))])
    Df = LinearOperator(space(U),space(U),zeros(dimension(space(U)),dimension(space(U))))
    Df = DFₚ!(Df,U,ν,ν₁,ν₂,ν₃)
    Df⁻¹ = inv(Df)
    U̇ = Df⁻¹*U̇
    Ẋn = zero(Ẋ)
    component(Ẋn,1)[1] = -1
    component(Ẋn,2) .= U̇
    Ẋn = Ẋn/norm(Ẋn)
    Σ = sum(component(Ẋn,2).*component(Ẋ,2))
    if (Σ > 0)
        return Ẋn
    else
        return -Ẋn
    end
end

function _newton_continueo(X,Ẋ,ν,ν₁,ν₂,ν₃,ν₄,ds,jmax)
    norm_U = []
    ν₅_list = []
    Xvec = []
    Ẋvec = []
    push!(Xvec,X)
    push!(Ẋvec,Ẋ)
    push!(ν₅_list,component(X,1)[1])
    push!(norm_U, norm(component(X,2),2))
    for j = 1:jmax
        Ẋ = TangentVector(X,Ẋ,ν,ν₁,ν₂,ν₃)
        X = X + ds*Ẋ
        U₀ = component(X,2)
        Y,ϵ = _newton_sc(X,U₀,component(Ẋ,2),ν,ν₁,ν₂,ν₃,ν₄)
        if ϵ > 7
            return Vector{Float64}(norm_U),Vector{Float64}(ν₅_list),Xvec,Ẋvec
        elseif ϵ == 15
            return Vector{Float64}(norm_U),Vector{Float64}(ν₅_list),Xvec,Ẋvec
        else
            X = Y
        end
        @show component(X,1)[1]
        J = component(X,2)
        push!(norm_U,L_2_norm(X))
        push!(ν₅_list,component(X,1)[1])
        d = π/frequency(J)[1]
        push!(Xvec,X)
        push!(Ẋvec,Ẋ)
        @show j
    end
    return Vector{Float64}(norm_U),Vector{Float64}(ν₅_list),Xvec,Ẋvec
end

function _newton_continue(X,Ẋ,ν,ν₁,ν₂,ν₃,ν₄,arclength_grid,npts)
    norm_U = []
    ν₅_list = []
    Xvec = []
    Ẋvec = []
    push!(Xvec,X)
    push!(Ẋvec,Ẋ)
    push!(ν₅_list,component(X,1)[1])
    push!(norm_U, norm(component(X,2),2))
    for j = 2:npts
        Ẋ = TangentVector(X,Ẋ,ν,ν₁,ν₂,ν₃)
        ds = arclength_grid[j] - arclength_grid[j-1]
        X = X + ds*Ẋ
        U₀ = component(X,2)
        Y,ϵ = _newton_sc(X,U₀,component(Ẋ,2),ν,ν₁,ν₂,ν₃,ν₄)
        if ϵ > 7
            return Vector{Float64}(norm_U),Vector{Float64}(ν₅_list),Xvec,Ẋvec
        elseif ϵ == 15
            return Vector{Float64}(norm_U),Vector{Float64}(ν₅_list),Xvec,Ẋvec
        else
            X = Y
        end
        @show component(X,1)[1]
        J = component(X,2)
        push!(norm_U,norm(X,1))
        push!(ν₅_list,component(X,1)[1])
        d = π/frequency(J)[1]
        push!(Xvec,X)
        push!(Ẋvec,Ẋ)
        @show j
    end
    return Vector{Float64}(norm_U),Vector{Float64}(ν₅_list),Xvec,Ẋvec
end

function _newton_sc(X,U₀,U̇,ν,ν₁,ν₂,ν₃,ν₄)
    T = similar(X)
    s = space(X)
    r = length(X)
    DT = LinearOperator(s,s,similar(coefficients(X),r,r))
    T .= 0 
    DT .= 0
    ϵ = 1
    l = 0
    while (ϵ > 1e-13) & (l < 15)
        T = F_con!(T,X,U₀,U̇,ν,ν₁,ν₂,ν₃,ν₄)
        DT = DF_con!(DT,X,U̇,ν,ν₁,ν₂,ν₃)
        X = X - DT\T
        ϵ = norm(T,Inf)
        nv = norm(X,Inf)
        if nv < 1e-5
            print("Newton may have converged to the zero solution")
        end
        if ϵ > 7
            @show opnorm(inv(DT),2)
            print("Newton may have diverged")
            return zero(X),ϵ
        end
        if (Int(l + 1) == 15)
            @show opnorm(inv(DT),2)
            print("Newton is having trouble converging")
            return zero(X),15
        end
        l += 1
    end
    return X,ϵ
end

# Checks the conditions of the Radii-Polynomial Theorem (see Section 4).
function CAP(Y₀,Z₁,Z₂,R)
    if Z₁ > 1
        display("Z₁ is too big")
        return Z₁
    elseif 2Y₀*Z₂ > (1-Z₁)^2
        display("The condition 2Y₀*Z₂ < (1-Z₁)² is not satisfied")
        return Y₀,Z₁,Z₂
    else
        r_min = (1 - Z₁ - sqrt((1-Z₁)^2 - 2Y₀*Z₂))/Z₂
        r_max = min((1-Z₁)/Z₂, R)
        if r_min < r_max
            display("The computer assisted proof was successful!")
            return r_min,r_max
        else 
            display("Choice of R was too small.")
        return r_min,r_max
        end
    end
end

# Takes a grid of ParameterSpace() × fourier^2 sequences and fits a Chebyshev sequence to it.
grid2cheb(x_fft, Nc) = 
    [rifft!(complex.(getindex.(x_fft, i)), Chebyshev(Nc)) for i ∈ indices(space(x_fft[1]))]

# Takes a grid of ParameterSpace() × fourier^2 operators and fits a Chebyshev sequence to it.
grid2chebm(x_fft, Nc) =
    [rifft!(complex.(getindex.(x_fft, i, j)), Chebyshev(Nc)) for i ∈ indices(codomain(x_fft[1])), j ∈ indices(domain(x_fft[1]))]

function grid2cheb(x_fft, Space, Nc)
    x_cheb = grid2cheb(x_fft,Nc)
    x_cheb0= [real.(getindex.(x_cheb, i)) for i ∈ eachindex(coefficients(x_cheb[1])).-1]
    x_cheb0 = [Sequence(Space,x_cheb0[i]) for i = 1:length(x_cheb0)]
    return x_cheb,x_cheb0
end

function grid2chebm(x_fft,Space1,Space2,Nc)
    x_cheb = grid2chebm(x_fft,Nc)
    x_cheb0mat =  [real.(getindex.(x_cheb, i)) for i ∈ eachindex(coefficients(x_cheb[1])).-1] 
    return x_cheb,LinearOperator.(Space1,Space2, x_cheb0mat)
end

# Takes a Chebyshev sequence and converts it back to a grid of ParameterSpace() × D₆Fourier sequences or operators.
function cheb2grid(x::VecOrMat{<:Sequence}, N_fft)
    vals = RadiiPolynomial.fft.(x, N_fft)
    return [real.(getindex.(vals, i)) for i ∈ eachindex(vals[1])]
end

function cheb2grid(x::VecOrMat{<:Sequence},Space, N_fft)
    vals = cheb2grid(x, N_fft)
    return Sequence.(Space, vals)
end

function _compute_Φ̄ₚ(W̄_cheb,Space,Nc,ν₂)
    # Φ̄ₚ(s) is a Chebyshev sequence of order 2Nc. 
    N2 = 2Nc 
    N2_fft = nextpow(2, 2N2 + 1)
    W_fft = cheb2grid(W̄_cheb,Space,N2_fft)
    U₁_fft = component.(component.(W_fft,2),1)
    Φ̄ₚ_fft = one.(U₁_fft) .+ U₁_fft .+ ν₂*U₁_fft.^2
    Φ̄ₚ_inv_fft = inv.(Φ̄ₚ_fft)
    Space2 = CosFourier(2order(U₁_fft[1]),frequency(U₁_fft[1]))
    Φ̄ₚ_cheb,Φ̄ₚ_cheb0 = grid2cheb(Φ̄ₚ_fft,Space2,N2)
    Φ̄ₚ_inv_cheb,Φ̄ₚ_inv_cheb0 = grid2cheb(Φ̄ₚ_inv_fft,Space2,N2)
    return Φ̄ₚ_cheb,Φ̄ₚ_cheb0,Φ̄ₚ_inv_cheb,Φ̄ₚ_inv_cheb0
end

function _compute_Ψ̄ₚ(W̄_cheb,Space,Nc,ν,ν₁)
    # Ψ̄ₚ is a Chebyshev sequence of size 2Nc 
    N2 = 2Nc 
    N2_fft = nextpow(2, 2N2 + 1)
    W_fft = cheb2grid(W̄_cheb,Space,N2_fft)
    Ū₁_fft = component.(component.(W_fft,2),1)
    Ū₂_fft = component.(component.(W_fft,2),2)
    Ψ̄ₚ_fft = -(ν₁*ν*Ū₁_fft.^2 .- ν₁*Ū₁_fft.*Ū₂_fft)
    Space2 = CosFourier(2order(Ū₁_fft[1]),frequency(Ū₁_fft[1]))
    Ψ̄ₚ_cheb,Ψ̄ₚ_cheb0 = grid2cheb(Ψ̄ₚ_fft,Space2,N2)
    return Ψ̄ₚ_cheb,Ψ̄ₚ_cheb0
end

function _compute_Ψ̄ₚ₁(W̄_cheb,Space,Nc,ν,ν₁,ν₂)
    # Ψ̄ₚ₁ is a Chebyshev sequence of size 3Nc 
    N3 = 3Nc 
    N3_fft = nextpow(2, 2N3 + 1)
    W_fft = cheb2grid(W̄_cheb,Space,N3_fft)
    Ū₁_fft = component.(component.(W_fft,2),1)
    Ū₂_fft = component.(component.(W_fft,2),2)
    Ψ̄ₚ₁_fft = -ν₁*(-Ū₂_fft .+ exact(2)*ν*Ū₁_fft .+ ν*Ū₁_fft.^2 .+ ν₂*Ū₁_fft.^2 .*Ū₂_fft)
    Space3 = CosFourier(3order(Ū₁_fft[1]),frequency(Ū₁_fft[1]))
    Ψ̄ₚ₁_cheb,Ψ̄ₚ₁_cheb0 = grid2cheb(Ψ̄ₚ₁_fft,Space3,N3)
    return Ψ̄ₚ₁_cheb,Ψ̄ₚ₁_cheb0
end

function _compute_gₚ(Ψ̄ₚ_cheb,Φ̄ₚ_inv_cheb,Space,Nc,ν₄)
    #gₚ is a Chebyshev sequence of order 4Nc
    N4 = 4Nc 
    N4_fft = nextpow(2, 2N4 + 1)
    Ψ̄ₚ_fft = cheb2grid(Ψ̄ₚ_cheb,Space,N4_fft)
    Φ̄ₚ_inv_fft = cheb2grid(Φ̄ₚ_inv_cheb,Space,N4_fft)
    gₚ_fft = one.(Ψ̄ₚ_fft)*ν₄ .+ Ψ̄ₚ_fft.*Φ̄ₚ_inv_fft
    Space2 = CosFourier(2order(Ψ̄ₚ_fft[1]),frequency(Ψ̄ₚ_fft[1]))
    gₚ_cheb,gₚ_cheb0 = grid2cheb(gₚ_fft,Space2,N4)
    return gₚ_cheb,gₚ_cheb0
end

function _compute_Vₚ₁(Ψ̄ₚ₁_cheb,Φ̄ₚ_inv_cheb,Space1,Space2,Nc,fourier)
    #Vₚ₁ is a Chebyshev sequence of order 7Nc
    N7 = 7Nc 
    N7_fft = nextpow(2, 2N7 + 1)
    Ψ̄ₚ₁_fft = cheb2grid(Ψ̄ₚ₁_cheb,Space1,N7_fft)
    Φ̄ₚ_inv_fft = cheb2grid(Φ̄ₚ_inv_cheb,Space2,N7_fft)
    Vₚ₁_fft = Ψ̄ₚ₁_fft.*Φ̄ₚ_inv_fft.^2 
    N = div(order(Space2),2)
    Vₚ₁ᴺ_fft = project.(Vₚ₁_fft,fourier)
    Space7 = CosFourier(7N,frequency(Space2))
    Vₚ₁_cheb,Vₚ₁_cheb0 = grid2cheb(Vₚ₁_fft,Space7,N7)
    Vₚ₁ᴺ_cheb,Vₚ₁ᴺ_cheb0 = grid2cheb(Vₚ₁ᴺ_fft,fourier,N7)
    return Vₚ₁_cheb,Vₚ₁_cheb0,Vₚ₁ᴺ_cheb,Vₚ₁ᴺ_cheb0
end

function _compute_Vₚ₂(Ψ̄ₚ₂_cheb,Φ̄ₚ_inv_cheb,Space1,Space2,Nc,fourier)
    #Vₚ₂ is a Chebyshev sequence of order 3Nc
    N3 = 3Nc 
    N3_fft = nextpow(2, 2N3 + 1)
    Ψ̄ₚ₂_fft = cheb2grid(Ψ̄ₚ₂_cheb,Space1,N3_fft)
    Φ̄ₚ_inv_fft = cheb2grid(Φ̄ₚ_inv_cheb,Space2,N3_fft)
    Vₚ₂_fft = Ψ̄ₚ₂_fft.*Φ̄ₚ_inv_fft
    Vₚ₂ᴺ_fft = project.(Vₚ₂_fft,fourier)
    N = order(Space1)
    Space3 = CosFourier(3N,frequency(Space1))
    Vₚ₂_cheb,Vₚ₂_cheb0 = grid2cheb(Vₚ₂_fft,Space3,N3)
    Vₚ₂ᴺ_cheb,Vₚ₂ᴺ_cheb0 = grid2cheb(Vₚ₂ᴺ_fft,fourier,N3)
    return Vₚ₂_cheb,Vₚ₂_cheb0,Vₚ₂ᴺ_cheb,Vₚ₂ᴺ_cheb0
end

# Allows us to switch between cos and exponential Fourier series.
function _build_P(τ,space)
    ord = order(space)[1]
    V = interval.(vec(zeros(dimension(space))))
    V[1] = interval(1)
    for k = 2:(ord+1)
        V[k] = exact(2)*τ^interval(k) 
    end
    return V
end

# Computes the operator norm on Chebyshev series as defined in Section 4.
function opnorm_cheb(s,τ)
    dom = domain(s[1])[2][1]
    codom = codomain(s[1])[2][1]
    P = _build_P(τ,codom)
    P⁻¹ = interval.(ones(dimension(dom)))./_build_P(τ,dom)
    P = [interval(1) ; P ; P]
    P⁻¹ = [interval(1) ; P⁻¹ ; P⁻¹]
    opnorm_ans = opnorm(P.*s[1].*P⁻¹',1)
    l = length(s)
    for i = 2:l 
        opnorm_ans += exact(2)*opnorm(P.*s[i].*P⁻¹',1)
    end 
    return opnorm_ans
end

function norm_cheb(s,X)
    norm_ans = norm(s[1],X)
    l = length(s)
    for i = 2:l 
        norm_ans += exact(2)*norm(s[i],X)
    end 
    return norm_ans
end

function _compute_Y_tail_terms3(A_con_cheb,Ψ̄ₚ_cheb,Φ̄ₚ_inv_cheb,Φ̄ₚ_cheb,Space,fourier,Nc)
    N4 = 4Nc 
    N4_fft = nextpow(2,2N4+1)
    
    Φ̄ₚ_fft_N4 = cheb2grid(Φ̄ₚ_cheb,Space,N4_fft)
    Φ̄ₚ_inv_fft_N4 = cheb2grid(Φ̄ₚ_inv_cheb,Space,N4_fft)
    denom_term_fft = one.(Φ̄ₚ_fft_N4) .- Φ̄ₚ_fft_N4.*Φ̄ₚ_inv_fft_N4
    N = div(order(Space),2)
    Space4 = CosFourier(4N,frequency(Space))
    denom_term_cheb,denom_term_cheb0 = grid2cheb(denom_term_fft,Space4,N4)
    #denom_term_cheb0 = _big_to_int0(denom_term_cheb0,Vector{Sequence{CosFourier{Interval{Float64}}, Vector{Interval{Float64}}}})
    
    N8 = 8Nc 
    N8_fft = nextpow(2,2N8+1)
    Ψ̄ₚ_fft_N8 = cheb2grid(Ψ̄ₚ_cheb,Space,N8_fft)
    Φ̄ₚ_fft_N8 = cheb2grid(Φ̄ₚ_cheb,Space,N8_fft)
    Φ̄ₚ_inv_fft_N8 = cheb2grid(Φ̄ₚ_inv_cheb,Space,N8_fft)
    num_term_fft =  Ψ̄ₚ_fft_N8.*Φ̄ₚ_inv_fft_N8.*(one.(Φ̄ₚ_fft_N8) .- Φ̄ₚ_fft_N8.*Φ̄ₚ_inv_fft_N8)
    Space8 = CosFourier(8N,frequency(Space))
    num_term_cheb,num_term_cheb0 = grid2cheb(num_term_fft - project.(num_term_fft,fourier),Space8,N8)
    #num_term_cheb0 = _big_to_int0(num_term_cheb0,Vector{Sequence{CosFourier{Interval{Float64}}, Vector{Interval{Float64}}}})

    N9 = 9Nc 
    N9_fft = nextpow(2,2N9+1)
    Ψ̄ₚ_fft_N9 = cheb2grid(Ψ̄ₚ_cheb,Space,N9_fft)
    Φ̄ₚ_fft_N9 = cheb2grid(Φ̄ₚ_cheb,Space,N9_fft)
    Φ̄ₚ_inv_fft_N9 = cheb2grid(Φ̄ₚ_inv_cheb,Space,N9_fft)
    _inside_part = project.(Ψ̄ₚ_fft_N9.*Φ̄ₚ_inv_fft_N9.*(one.(Φ̄ₚ_fft_N9) .- Φ̄ₚ_fft_N9.*Φ̄ₚ_inv_fft_N9),fourier)
    #_inside_part = _big_to_int0(_inside_part,Vector{Sequence{CosFourier{Interval{Float64}}, Vector{Interval{Float64}}}})
    pSpace = ParameterSpace() × fourier^2 
    _p_inside_part = Vector{Sequence{CartesianProduct{Tuple{ParameterSpace, CartesianPower{CosFourier{Interval{Float64}}}}}, Vector{Interval{Float64}}}}(undef,N9_fft)
    for k = 1:N9_fft
        _p_inside_part[k] = Sequence(pSpace, [interval(0) ; coefficients(_inside_part[k]) ; interval.(zeros(length(_inside_part[k])))])
    end 
    Anum_term_fft = Sequence.(pSpace, cheb2grid(A_con_cheb,N9_fft).* coefficients.(_p_inside_part))
    Anum_term_cheb,Anum_term_cheb0 = grid2cheb(Anum_term_fft,pSpace,N9)
    return Anum_term_cheb0,num_term_cheb0,denom_term_cheb0
end

function _Z₂₃_term_cheb(W̄_cheb,Space,Nc,ν₂)
    #Ū₁² is a Chebyshev sequence of size 2Nc 
    N2 = 2Nc 
    N2_fft = nextpow(2,2N2 + 1)
    W_fft = cheb2grid(W̄_cheb,Space,N2_fft)
    Ū₁_fft = component.(component.(W_fft,2),1)
    _this_term = one.(Ū₁_fft) .- ν₂*Ū₁_fft.^2
    N = order(Space[2])[1]
    Space2 = CosFourier(2N,frequency(Space[2])[1])
    _this_term,_this_term0 = grid2cheb(_this_term,Space2,N2)
    return _this_term0 
end

function _Z₂₁_term1_cheb(W̄_cheb,Space,Nc,ν,ν₁,ν₂)
    #This term is a Chebyshev sequence of size 4Nc 
    N4 = 4Nc 
    N4_fft = nextpow(2,2N4 + 1)
    W_fft = cheb2grid(W̄_cheb,Space,N4_fft)
    Ū₁_fft = component.(component.(W_fft,2),1)
    Ū₂_fft = component.(component.(W_fft,2),2)
    _this_term = ν₂^2*Ū₁_fft.^3 .*Ū₂_fft .+ ν*ν₂.*Ū₁_fft.^3 .+ exact(3)*ν*ν₂.*Ū₁_fft.^2 - exact(3)*ν₂.*Ū₁_fft.*Ū₂_fft .-ν*one.(Ū₁_fft) .-Ū₂_fft
    N = order(Space[2])[1]
    Space4 = CosFourier(4N,frequency(Space[2])[1])
    _this_term,_this_term0 = grid2cheb(_this_term,Space4,N4)
    return _this_term0 
end

function _Z₂₁_term2_and3_cheb(W̄_cheb,Space,Nc,ν,ν₁,ν₂,ν₃)
    #This term is a Chebyshev sequence of size 3Nc 
    N3 = 3Nc 
    N3_fft = nextpow(2,2N3 + 1)
    W_fft = cheb2grid(W̄_cheb,Space,N3_fft)
    Ū₁_fft = component.(component.(W_fft,2),1)
    Ū₂_fft = component.(component.(W_fft,2),2)
    _this_term2 = -exact(1)*one.(Ū₁_fft) .-exact(3)*ν₂*Ū₁_fft .+ ν₂*Ū₁_fft.^3
    _this_term3 = exact(6)*ν*ν₂*Ū₁_fft .+ exact(3)*ν*ν₂*Ū₁_fft.^2 .- exact(3)*ν₂*Ū₂_fft .+ exact(3)*ν₂^2*Ū₁_fft.^2 .*Ū₂_fft
    N = order(Space[2])[1]
    Space3 = CosFourier(3N,frequency(Space[2])[1])
    _this_term2,_this_term20 = grid2cheb(_this_term2,Space3,N3)
    _this_term3,_this_term30 = grid2cheb(_this_term3,Space3,N3)
    return _this_term20,_this_term30
end

function _Z₂₁_term4_and5_cheb(W̄_cheb,Space,Nc,ν,ν₂)
    #This term is a Chebyshev sequence of size 2Nc 
    N2 = 2Nc 
    N2_fft = nextpow(2,2N2 + 1)
    W_fft = cheb2grid(W̄_cheb,Space,N2_fft)
    Ū₁_fft = component.(component.(W_fft,2),1)
    _this_term4 = -exact(3)*ν₂*one.(Ū₁_fft) .+ exact(3)*ν₂^2*Ū₁_fft.^2
    _this_term5 = exact(3)*ν*ν₂*one.(Ū₁_fft) .+exact(3)*ν*ν₂*Ū₁_fft .+exact(3)*ν₂^2*Ū₁_fft.^2
    N = order(Space[2])[1]
    Space2 = CosFourier(2N,frequency(Space[2])[1])
    _this_term4,_this_term40 = grid2cheb(_this_term4,Space2,N2)
    _this_term5,_this_term50 = grid2cheb(_this_term5,Space2,N2)
    return _this_term40,_this_term50
end

function _compute_Z_infty_tail_terms2(Ψ̄ₚ₁_cheb,Ψ̄ₚ₂_cheb,Φ̄ₚ_inv_cheb,Φ̄ₚ_cheb,Space,Nc)
    N8 = 8Nc 
    N8_fft = nextpow(2,2N8+1)
    
    Φ̄ₚ_fft_N8 = cheb2grid(Φ̄ₚ_cheb,Space,N8_fft)
    Φ̄ₚ_inv_fft_N8 = cheb2grid(Φ̄ₚ_inv_cheb,Space,N8_fft)
    denom_term_fft = one.(Φ̄ₚ_fft_N8) .- Φ̄ₚ_fft_N8.^2 .*Φ̄ₚ_inv_fft_N8.^2
    N = div(order(Space),2)
    Space8 = CosFourier(8N,frequency(Space))
    denom_term_cheb,denom_term_cheb0 = grid2cheb(denom_term_fft,Space8,N8)
    N15 = 15Nc 
    N15_fft = nextpow(2,2N15+1)
    Φ̄ₚ_fft_N15 = cheb2grid(Φ̄ₚ_cheb,Space,N15_fft)
    Φ̄ₚ_inv_fft_N15 = cheb2grid(Φ̄ₚ_inv_cheb,Space,N15_fft)
    Ψ̄ₚ₁_fft_N15 = cheb2grid(Ψ̄ₚ₁_cheb,CosFourier(3N,frequency(Space)),N15_fft)
    num_term1_fft = Ψ̄ₚ₁_fft_N15.*Φ̄ₚ_inv_fft_N15.^2 .*(one.(Φ̄ₚ_fft_N15) .- Φ̄ₚ_fft_N15.^2 .*Φ̄ₚ_inv_fft_N15.^2)
    Space15 = CosFourier(15N,frequency(Space))
    num_term1_cheb,num_term1_cheb0 = grid2cheb(num_term1_fft,Space15,N15)

    N7 = 7Nc 
    N7_fft = nextpow(2,2N7+1)
    Φ̄ₚ_fft_N7 = cheb2grid(Φ̄ₚ_cheb,Space,N7_fft)
    Φ̄ₚ_inv_fft_N7 = cheb2grid(Φ̄ₚ_inv_cheb,Space,N7_fft)
    Ψ̄ₚ₂_fft_N7 = cheb2grid(Ψ̄ₚ₂_cheb,CosFourier(N,frequency(Space)),N7_fft)
    num_term2_fft = Ψ̄ₚ₂_fft_N7.*Φ̄ₚ_inv_fft_N7 .*(one.(Φ̄ₚ_fft_N7) .- Φ̄ₚ_fft_N7 .*Φ̄ₚ_inv_fft_N7)
    Space7 = CosFourier(7N,frequency(Space))
    num_term2_cheb,num_term2_cheb0 = grid2cheb(num_term2_fft,Space7,N7)
    return num_term1_cheb0,num_term2_cheb0,denom_term_cheb0 
end

function _subtract_from_identity(M)
    l = length(M)
    for k = 1:l
        M[k] = UniformScaling(interval(1)) - M[k] 
    end
    return M
end

function _big_to_int(a_big_cheb)
    l = length(a_big_cheb)
    a_cheb = Vector{Sequence{Chebyshev, Vector{Interval{Float64}}}}(undef,l)
    for k = 1:l 
        a_cheb[k] = interval.(Float64.(inf.(a_big_cheb[k] ),RoundDown),Float64.(sup.(a_big_cheb[k] ),RoundUp) )
    end
    return a_cheb 
end

function _big_to_int0(a_big_cheb0,T)
    l = length(a_big_cheb0)
    a_cheb0 = T(undef,l)
    for k = 1:l 
        a_cheb0[k] = interval.(Float64.(inf.(a_big_cheb0[k] ),RoundDown),Float64.(sup.(a_big_cheb0[k] ),RoundUp) )
    end
    return a_cheb0 
end

#################################################### Main Code ####################################################
# Segment 1
N = 80
N₀ = 90
# Defining the parameters
ν = 0.42^2 ; νi = interval(ν) ; νbig = interval(big(ν))
ν₄ = 21 ; ν₄i = interval(ν₄) ; ν₄big = interval(big(ν₄))
ν₃ = 0.28 ; ν₃i = interval(ν₃) ; ν₃big = interval(big(ν₃))
ν₁ = 8 ; ν₁i = interval(ν₁) ; ν₁big = interval(big(ν₁))
ν₂ = 1 ; ν₂i = interval(ν₂) ; ν₂big = interval(big(ν₂))
d = 5 ; di = interval(d) ; dbig = interval(big(d))
τ = interval(1.0) 
R = interval(9e-7)
fourier0 = CosFourier(N₀,π/di)
fourier = CosFourier(N,π/di)
fourier_mid = CosFourier(N,π/d)
fourier0_mid = CosFourier(N₀,π/d)
pfourier_mid = ParameterSpace() × fourier_mid^2
pfourier0_mid = ParameterSpace() × fourier0_mid^2
pfourier = ParameterSpace() × fourier^2
pfourier0 = ParameterSpace() × fourier0^2
# Building continuation objects
Nc = 127
N_fft = nextpow(2, 2Nc + 1)
npts = N_fft ÷ 2 + 1

arclength = -0.08
arclength_grid = [0.5 * arclength - 0.5 * cospi(2j/N_fft) * arclength for j ∈ 0:npts-1]


U = load("Ubar_Thomas_Branch_start","U")
ν₅ = 67.59937678730851
U₁ = component(U,1)
U₂ = ν*U₁ - component(U,2)
U = Sequence(CosFourier(N₀,π/d)^2, [coefficients(project(U₁,CosFourier(N₀,π/d))) ; coefficients(project(U₂,CosFourier(N₀,π/d)))])
Ū,err = _newton_s(U,15,ν,ν₁,ν₂,ν₃,ν₄,ν₅)

W̄ = Sequence(pfourier0_mid, [ν₅ ; coefficients(Ū)])
Ẇ = Sequence(pfourier0_mid, zeros(1+2*(N₀+1)))
Ẇ = TangentVector(W̄,Ẇ,ν,ν₁,ν₂,ν₃)
norm_U,b_list,W_grid,Ẇ_grid = _newton_continue(W̄,Ẇ,ν,ν₁,ν₂,ν₃,ν₄,arclength_grid,npts)


#W̄_cheb_prev = W̄_cheb
#R_prev = vals[1]
#= Segment 2
N = 100
N₀ = 100
R = interval(1e-6)
fourier0 = CosFourier(N₀,π/di)
fourier = CosFourier(N,π/di)
fourier_mid = CosFourier(N,π/d)
fourier0_mid = CosFourier(N₀,π/d)
pfourier_mid = ParameterSpace() × fourier_mid^2
pfourier0_mid = ParameterSpace() × fourier0_mid^2
pfourier = ParameterSpace() × fourier^2
pfourier0 = ParameterSpace() × fourier0^2
# Building continuation objects
Nc = 63
N_fft = nextpow(2, 2Nc + 1)
npts = N_fft ÷ 2 + 1

W̄ = project(W_grid[end],pfourier0_mid)
Ẇ = project(Ẇ_grid[end],pfourier0_mid)

arclength2 = -0.02
arclength_grid2 = [0.5 * arclength2 - 0.5 * cospi(2j/N_fft) * arclength2 for j ∈ 0:npts-1]

norm_U2,b_list2,W_grid,Ẇ_grid = _newton_continue(W̄,Ẇ,ν,ν₁,ν₂,ν₃,ν₄,arclength_grid2,npts)


# Segment 3 
N = 120
N₀ = 120
R = interval(7e-7)
fourier0 = CosFourier(N₀,π/di)
fourier = CosFourier(N,π/di)
fourier_mid = CosFourier(N,π/d)
fourier0_mid = CosFourier(N₀,π/d)
pfourier_mid = ParameterSpace() × fourier_mid^2
pfourier0_mid = ParameterSpace() × fourier0_mid^2
pfourier = ParameterSpace() × fourier^2
pfourier0 = ParameterSpace() × fourier0^2
# Building continuation objects
Nc = 63
N_fft = nextpow(2, 2Nc + 1)
npts = N_fft ÷ 2 + 1

W̄ = project(W_grid[end],pfourier0_mid)
Ẇ = project(Ẇ_grid[end],pfourier0_mid)
arclength3 = -0.2
arclength_grid3 = [0.5 * arclength3 - 0.5 * cospi(2j/N_fft) * arclength3 for j ∈ 0:npts-1]

norm_U3,b_list3,W_grid,Ẇ_grid = _newton_continue(W̄,Ẇ,ν,ν₁,ν₂,ν₃,ν₄,arclength_grid3,npts)

# Segment 4 
N = 110
N₀ = 110
R = interval(9e-7)
fourier0 = CosFourier(N₀,π/di)
fourier = CosFourier(N,π/di)
fourier_mid = CosFourier(N,π/d)
fourier0_mid = CosFourier(N₀,π/d)
pfourier_mid = ParameterSpace() × fourier_mid^2
pfourier0_mid = ParameterSpace() × fourier0_mid^2
pfourier = ParameterSpace() × fourier^2
pfourier0 = ParameterSpace() × fourier0^2
# Building continuation objects
Nc = 127
N_fft = nextpow(2, 2Nc + 1)
npts = N_fft ÷ 2 + 1

W̄ = project(W_grid[end],pfourier0_mid)
Ẇ = project(Ẇ_grid[end],pfourier0_mid)
arclength4 = -0.3
arclength_grid4 = [0.5 * arclength4 - 0.5 * cospi(2j/N_fft) * arclength4 for j ∈ 0:npts-1]

norm_U4,b_list4,W_grid,Ẇ_grid = _newton_continue(W̄,Ẇ,ν,ν₁,ν₂,ν₃,ν₄,arclength_grid4,npts)

# Segment 5 
N = 90
N₀ = 100
R = interval(7e-7)
fourier0 = CosFourier(N₀,π/di)
fourier = CosFourier(N,π/di)
fourier_mid = CosFourier(N,π/d)
fourier0_mid = CosFourier(N₀,π/d)
pfourier_mid = ParameterSpace() × fourier_mid^2
pfourier0_mid = ParameterSpace() × fourier0_mid^2
pfourier = ParameterSpace() × fourier^2
pfourier0 = ParameterSpace() × fourier0^2
# Building continuation objects
Nc = 63
N_fft = nextpow(2, 2Nc + 1)
npts = N_fft ÷ 2 + 1

W̄ = project(W_grid[end],pfourier0_mid)
Ẇ = project(Ẇ_grid[end],pfourier0_mid)
arclength5 = -1.5
arclength_grid5 = [0.5 * arclength5 - 0.5 * cospi(2j/N_fft) * arclength5 for j ∈ 0:npts-1]

norm_U5,b_list5,W_grid,Ẇ_grid = _newton_continue(W̄,Ẇ,ν,ν₁,ν₂,ν₃,ν₄,arclength_grid5,npts)

# Segment 5_5 
N = 160
N₀ = 160
R = interval(7e-7)
fourier0 = CosFourier(N₀,π/di)
fourier = CosFourier(N,π/di)
fourier_mid = CosFourier(N,π/d)
fourier0_mid = CosFourier(N₀,π/d)
pfourier_mid = ParameterSpace() × fourier_mid^2
pfourier0_mid = ParameterSpace() × fourier0_mid^2
pfourier = ParameterSpace() × fourier^2
pfourier0 = ParameterSpace() × fourier0^2
# Building continuation objects
Nc = 63
N_fft = nextpow(2, 2Nc + 1)
npts = N_fft ÷ 2 + 1

W̄ = project(W_grid[end],pfourier0_mid)
Ẇ = project(Ẇ_grid[end],pfourier0_mid)
arclength5_5 = -3.5
arclength_grid5_5 = [0.5 * arclength5_5 - 0.5 * cospi(2j/N_fft) * arclength5_5 for j ∈ 0:npts-1]

norm_U5_5,b_list5_5,W_grid,Ẇ_grid = _newton_continue(W̄,Ẇ,ν,ν₁,ν₂,ν₃,ν₄,arclength_grid5_5,npts)

# Segment 6 
N = 150
N₀ = 150
R = interval(2e-7)
fourier0 = CosFourier(N₀,π/di)
fourier = CosFourier(N,π/di)
fourier_mid = CosFourier(N,π/d)
fourier0_mid = CosFourier(N₀,π/d)
pfourier_mid = ParameterSpace() × fourier_mid^2
pfourier0_mid = ParameterSpace() × fourier0_mid^2
pfourier = ParameterSpace() × fourier^2
pfourier0 = ParameterSpace() × fourier0^2
# Building continuation objects
Nc = 63
N_fft = nextpow(2, 2Nc + 1)
npts = N_fft ÷ 2 + 1

W̄ = project(W_grid[end],pfourier0_mid)
Ẇ = project(Ẇ_grid[end],pfourier0_mid)
arclength6 = -2.2
arclength_grid6 = [0.5 * arclength6 - 0.5 * cospi(2j/N_fft) * arclength6 for j ∈ 0:npts-1]

norm_U6,b_list6,W_grid,Ẇ_grid = _newton_continue(W̄,Ẇ,ν,ν₁,ν₂,ν₃,ν₄,arclength_grid6,npts)

# Segment 6_5
N = 100
N₀ = 110
R = interval(4e-7)
fourier0 = CosFourier(N₀,π/di)
fourier = CosFourier(N,π/di)
fourier_mid = CosFourier(N,π/d)
fourier0_mid = CosFourier(N₀,π/d)
pfourier_mid = ParameterSpace() × fourier_mid^2
pfourier0_mid = ParameterSpace() × fourier0_mid^2
pfourier = ParameterSpace() × fourier^2
pfourier0 = ParameterSpace() × fourier0^2
# Building continuation objects
Nc = 31
N_fft = nextpow(2, 2Nc + 1)
npts = N_fft ÷ 2 + 1

W̄ = project(W_grid[end],pfourier0_mid)
Ẇ = project(Ẇ_grid[end],pfourier0_mid)
arclength6_5 = -1.2
arclength_grid6_5 = [0.5 * arclength6_5 - 0.5 * cospi(2j/N_fft) * arclength6_5 for j ∈ 0:npts-1]

norm_U6_5,b_list6_5,W_grid,Ẇ_grid = _newton_continue(W̄,Ẇ,ν,ν₁,ν₂,ν₃,ν₄,arclength_grid6_5,npts)

# Segment 7 
N = 90
N₀ = 90
R = interval(5e-7)
fourier0 = CosFourier(N₀,π/di)
fourier = CosFourier(N,π/di)
fourier_mid = CosFourier(N,π/d)
fourier0_mid = CosFourier(N₀,π/d)
pfourier_mid = ParameterSpace() × fourier_mid^2
pfourier0_mid = ParameterSpace() × fourier0_mid^2
pfourier = ParameterSpace() × fourier^2
pfourier0 = ParameterSpace() × fourier0^2
# Building continuation objects
Nc = 63
N_fft = nextpow(2, 2Nc + 1)
npts = N_fft ÷ 2 + 1

W̄ = project(W_grid[end],pfourier0_mid)
Ẇ = project(Ẇ_grid[end],pfourier0_mid)
arclength7 = -0.61
arclength_grid7 = [0.5 * arclength7 - 0.5 * cospi(2j/N_fft) * arclength7 for j ∈ 0:npts-1]

norm_U7,b_list7,W_grid,Ẇ_grid = _newton_continue(W̄,Ẇ,ν,ν₁,ν₂,ν₃,ν₄,arclength_grid7,npts)

# Segment 8 
N = 90
N₀ = 90
R = interval(5e-7)
fourier0 = CosFourier(N₀,π/di)
fourier = CosFourier(N,π/di)
fourier_mid = CosFourier(N,π/d)
fourier0_mid = CosFourier(N₀,π/d)
pfourier_mid = ParameterSpace() × fourier_mid^2
pfourier0_mid = ParameterSpace() × fourier0_mid^2
pfourier = ParameterSpace() × fourier^2
pfourier0 = ParameterSpace() × fourier0^2
# Building continuation objects
Nc = 15
N_fft = nextpow(2, 2Nc + 1)
npts = N_fft ÷ 2 + 1

W̄ = project(W_grid[end],pfourier0_mid)
Ẇ = project(Ẇ_grid[end],pfourier0_mid)
arclength8 = -0.19
arclength_grid8 = [0.5 * arclength7 - 0.5 * cospi(2j/N_fft) * arclength8 for j ∈ 0:npts-1]

norm_U8,b_list8,W_grid,Ẇ_grid = _newton_continue(W̄,Ẇ,ν,ν₁,ν₂,ν₃,ν₄,arclength_grid8,npts)
=#
#=
mat"
figure
hold on 
plot($b_list,$norm_U,'b.','MarkerSize',6) 
 plot($b_list2,$norm_U2,'b.','MarkerSize',6) 
  plot($b_list3,$norm_U3,'b.','MarkerSize',6)
  plot($b_list4,$norm_U4,'b.','MarkerSize',6)
  plot($b_list5,$norm_U5,'b.','MarkerSize',6)
  plot($b_list5_5,$norm_U5_5,'b.','MarkerSize',6)
  plot($b_list6,$norm_U6,'b.','MarkerSize',6)
  plot($b_list6_5,$norm_U6_5,'b.','MarkerSize',6)
  plot($b_list7,$norm_U7,'b.','MarkerSize',6)
  plot($b_list8,$norm_U8,'b.','MarkerSize',6)"
=#
W_fft = [W_grid ; reverse(W_grid)[2:npts-1]]
Ẇ_fft = [Ẇ_grid ; reverse(Ẇ_grid)[2:npts-1]]
W̄_cheb,W̄_cheb0 = (grid2cheb(W_fft,pfourier0_mid,Nc))
Ẇ_cheb,Ẇ_cheb0 = (grid2cheb(Ẇ_fft,pfourier0_mid,Nc))

setprecision(128)
W̄_big_cheb = interval.(big.(W̄_cheb))
W̄_big_cheb0 = interval.(big.(W̄_cheb0))
W̄_cheb = _big_to_int(W̄_big_cheb)
W̄_cheb0 = _big_to_int0(W̄_big_cheb0,Vector{Sequence{CartesianProduct{Tuple{ParameterSpace, CartesianPower{CosFourier{Interval{Float64}}}}}, Vector{Interval{Float64}}}})


ℓ¹_τ = Ell1(GeometricWeight(τ))
𝒳_τ = NormedCartesianSpace((ℓ¹_τ,ℓ¹_τ),ℓ¹())
X₁ = NormedCartesianSpace((ℓ¹(),𝒳_τ),ℓ¹())

################## Computing the nonlinear terms ######################################################
@info "Computing the nonlinear terms"
Ū_big_fft = component.(interval.(big.(W_fft)),2)
Ū₁_big_fft = component.(Ū_big_fft,1)
Ū₂_big_fft = component.(Ū_big_fft,2)
Φ̄ₚ_big_cheb,Φ̄ₚ_big_cheb0,Φ̄ₚ_inv_big_cheb,Φ̄ₚ_inv_big_cheb0 = _compute_Φ̄ₚ(W̄_big_cheb,pfourier0,Nc,ν₂big)
Φ̄ₚ_cheb = _big_to_int(Φ̄ₚ_big_cheb)
Φ̄ₚ_inv_cheb= _big_to_int(Φ̄ₚ_inv_big_cheb)
Φ̄ₚ_inv_cheb0 = _big_to_int0(Φ̄ₚ_inv_big_cheb0,Vector{Sequence{CosFourier{Interval{Float64}}, Vector{Interval{Float64}}}})

Ψ̄ₚ₂_big_cheb,Ψ̄ₚ₂_big_cheb0 = grid2cheb(ν₁big*Ū₁_big_fft,fourier0,Nc)
Ψ̄ₚ₂_cheb = _big_to_int(Ψ̄ₚ₂_big_cheb)
Ψ̄ₚ_big_cheb,Ψ̄ₚ_cheb0 = _compute_Ψ̄ₚ(W̄_big_cheb,pfourier0,Nc,νbig,ν₁big)
Ψ̄ₚ_cheb = _big_to_int(Ψ̄ₚ_big_cheb)
Ψ̄ₚ₁_big_cheb,Ψ̄ₚ₁_big_cheb0 = _compute_Ψ̄ₚ₁(W̄_big_cheb,pfourier0,Nc,νbig,ν₁big,ν₂big)
Ψ̄ₚ₁_cheb = _big_to_int(Ψ̄ₚ₁_big_cheb)
# Building gₚ, Vₚ₁, and Vₚ₂. Note that these are not exact as we cannot represent the full inverse. We name them for demonstrational purposes.
gₚ_big_cheb,gₚ_big_cheb0 = _compute_gₚ(Ψ̄ₚ_big_cheb,Φ̄ₚ_inv_big_cheb,CosFourier(2N₀,π/di),Nc,ν₄big) 
gₚ_cheb = _big_to_int(gₚ_big_cheb)
Vₚ₁_cheb,Vₚ₁_cheb0,Vₚ₁ᴺ_cheb,Vₚ₁ᴺ_cheb0 = _compute_Vₚ₁(Ψ̄ₚ₁_cheb,Φ̄ₚ_inv_cheb,CosFourier(3N₀,π/di),CosFourier(2N₀,π/di),Nc,fourier)
Vₚ₂_cheb,Vₚ₂_cheb0,Vₚ₂ᴺ_cheb,Vₚ₂ᴺ_cheb0 = _compute_Vₚ₂(Ψ̄ₚ₂_cheb,Φ̄ₚ_inv_cheb,CosFourier(N₀,π/di),CosFourier(2N₀,π/di),Nc,fourier)

@info "Computing the operator A_con"
A_con_grid = inv.(DF_con.(project.(W_fft,pfourier_mid),component.(project.((Ẇ_fft),pfourier_mid),2),ν,ν₁,ν₂,ν₃))
A_con_cheb,A_con_cheb0 = (grid2chebm(A_con_grid,pfourier,pfourier, Nc))
A_con_cheb = interval.(A_con_cheb)
A_con_cheb0 = interval.(A_con_cheb0)
print("Computing norm of A_con_cheb")
norm_A_con_cheb = opnorm_cheb(A_con_cheb0,τ)
@show norm_A_con_cheb
ℒ_∞ = exact(1)/abs(νi*(interval(N+1)*π/di)^2 + exact(1)) + abs((ν₃i*νi-exact(1))/((νi*(interval(N+1)*π/di)^2 + exact(1))*((interval(N+1)*π/di)^2 + ν₃i))) +exact(1)/abs((interval(N+1)*π/di)^2 + ν₃i)

################## Y₀ BOUND ######################################################
@info "Computing Y₀"
# A_conᴺ(s) * F_con(W̄(s)) is a Chebyshev sequence of order 5Nc. 
N5 = 5Nc 
N5_fft = nextpow(2,2N5 + 1)
F_big_con_W̄ = F_con(W̄_big_cheb,Ẇ_cheb,gₚ_big_cheb,νbig,ν₁big,ν₂big,ν₃big,ν₄big,N,N₀,di,N5)
F_con_W̄ = _big_to_int0(F_big_con_W̄,Vector{Sequence{CartesianProduct{Tuple{ParameterSpace, CartesianPower{CosFourier{Interval{Float64}}}}}, Vector{Interval{Float64}}}})
A_conF_con_W̄_fft = Sequence.(pfourier, cheb2grid(A_con_cheb,N5_fft).*coefficients.(F_con_W̄))
A_conF_con_W̄_cheb,A_conF_con_W̄_cheb0 = grid2cheb(A_conF_con_W̄_fft, pfourier,N5)

tail_Y01,tail_Y02 = _Y0_con_tail(W̄_cheb,gₚ_cheb,N,N₀,di,Nc,νi,ν₃i)
Y₀₁ = norm_cheb(A_conF_con_W̄_cheb0,X₁) + ℒ_∞*(norm_cheb(tail_Y01,ℓ¹_τ) + norm_cheb(tail_Y02,ℓ¹_τ))
@show Y₀₁

Anum_term_cheb0,num_term_cheb0,denom_term_cheb0 = _compute_Y_tail_terms3(A_con_cheb,Ψ̄ₚ_cheb,Φ̄ₚ_inv_cheb,Φ̄ₚ_cheb,CosFourier(2N₀,π/di),fourier,Nc)
Y₀₂ = (norm_cheb(Anum_term_cheb0,X₁) + ℒ_∞*norm_cheb(num_term_cheb0,ℓ¹_τ))/(exact(1)-norm_cheb(denom_term_cheb0,ℓ¹_τ))
Y₀ = Y₀₁ + Y₀₂
@show Y₀

################## 𝒵₂ BOUND ######################################################
@info "Computing Z₂"    
Ū₁_fft = _big_to_int0(Ū₁_big_fft,Vector{Sequence{CosFourier{Interval{Float64}}, Vector{Interval{Float64}}}})
Ū₂_fft = _big_to_int0(Ū₂_big_fft,Vector{Sequence{CosFourier{Interval{Float64}}, Vector{Interval{Float64}}}})
Ū₁_cheb0 = component.(component.(W̄_cheb0,2),1)
Z₂₁_term2,Z₂₁_term3 = _Z₂₁_term2_and3_cheb(W̄_cheb,pfourier0,Nc,νi,ν₁i,ν₂i,ν₃i)
Z₂₁_term4,Z₂₁_term5 = _Z₂₁_term4_and5_cheb(W̄_cheb,pfourier0,Nc,νi,ν₂i)
Z₂₁_term7 = grid2cheb(νi*ν₂i*one.(Ū₁_fft) .+ ν₂i*Ū₂_fft,fourier0,Nc)[2]
Z₂₁ = norm_cheb(_Z₂₁_term1_cheb(W̄_cheb,pfourier0,Nc,νi,ν₁i,ν₂i),ℓ¹_τ) + (norm_cheb(Z₂₁_term2,ℓ¹_τ) + norm_cheb(Z₂₁_term3,ℓ¹_τ))*R + (norm_cheb(Z₂₁_term4,ℓ¹_τ) + norm_cheb(Z₂₁_term5,ℓ¹_τ))*R^2 + (norm_cheb(exact(3)*ν₂i*Ū₁_cheb0 ,ℓ¹_τ) + norm_cheb(Z₂₁_term7,ℓ¹_τ))*R^3 + ν₂i^2*R^4
Z₂₂ = norm_cheb(Φ̄ₚ_inv_cheb0,ℓ¹_τ)^3/(exact(1) - norm_cheb(denom_term_cheb0,ℓ¹_τ) - R*norm_cheb(Φ̄ₚ_inv_cheb0,ℓ¹_τ))^3
Z₂₃ = (norm_cheb(_Z₂₃_term_cheb(W̄_cheb,pfourier0,Nc,ν₂i),ℓ¹_τ) + exact(2)*ν₂i*norm_cheb(Ū₁_cheb0,ℓ¹_τ)*R + ν₂i*R^2)*norm_cheb(Φ̄ₚ_inv_cheb0,ℓ¹_τ)^2/(exact(1) - norm_cheb(denom_term_cheb0,ℓ¹_τ) - R*norm_cheb(Φ̄ₚ_inv_cheb0,ℓ¹_τ))^2
Z₂ = exact(2)*ν₁i*(norm_A_con_cheb + ℒ_∞)*(Z₂₁*Z₂₂ + Z₂₃)
@show Z₂

################## 𝒵₁ BOUND ###################################################### 
#These are the actual Vₚⱼᴺ for j = 1,2.
Ẇ_cheb0 = Sequence.(pfourier0, coefficients.(interval.(Ẇ_cheb0)))
Ẇ_fft = Sequence.(pfourier0, coefficients.(interval.(Ẇ_fft)))
U̇_cheb0 = component.(Ẇ_cheb0,2)
U̇ᴺ_fft = project.(project.(component.(Ẇ_fft,2),fourier^2),fourier0^2)
U̇ᴺ_cheb,U̇ᴺ_cheb0 = (grid2cheb(U̇ᴺ_fft,fourier0^2,Nc))

num_term1_cheb0,num_term2_cheb0,denom_term2_cheb0 = _compute_Z_infty_tail_terms2(Ψ̄ₚ₁_cheb,Ψ̄ₚ₂_cheb,Φ̄ₚ_inv_cheb,Φ̄ₚ_cheb,CosFourier(2N₀,π/di),Nc)
Z_∞₁ = norm_cheb(U̇_cheb0 - U̇ᴺ_cheb0,𝒳_τ)
Z_∞₂ = norm_cheb(Vₚ₁_cheb0 - Vₚ₁ᴺ_cheb0,ℓ¹_τ) 
Z_∞₃ = norm_cheb(num_term1_cheb0,ℓ¹_τ)/(exact(1) - norm_cheb(denom_term2_cheb0,ℓ¹_τ))
Z_∞₄ = norm_cheb(Vₚ₂_cheb0 - Vₚ₂ᴺ_cheb0,ℓ¹_τ) + norm_cheb(num_term2_cheb0,ℓ¹_τ)/(exact(1)-norm_cheb(denom_term_cheb0,ℓ¹_τ))
Z_∞ = (norm_A_con_cheb + ℒ_∞)*(Z_∞₁ + Z_∞₂ + Z_∞₃ + Z_∞₄)

#We only projected down in Fourier, so A_conᴺ(s)*DF_con(W̄(s)) is a Chebyshev polynomial of order 8Nc (due to Vₚ₁ being of order 7Nc.)
N8 = 8Nc
N8_fft = nextpow(2,2N8+1)
W_fft_N8 = cheb2grid(W̄_cheb,pfourier0,N8_fft)
Ẇ_fft_N8 = cheb2grid(interval.(Ẇ_cheb),pfourier0,N8_fft)
Vₚ₁ᴺ_fft_N8 = cheb2grid(Vₚ₁ᴺ_cheb,fourier,N8_fft)
Vₚ₂ᴺ_fft_N8 = cheb2grid(Vₚ₂ᴺ_cheb,fourier,N8_fft)
DF_con_W̄_fft = DF_con_2N.(project.(W_fft_N8,pfourier),component.(project.(Ẇ_fft_N8,pfourier),2),νi,ν₁i,ν₂i,ν₃i,Vₚ₁ᴺ_fft_N8,Vₚ₂ᴺ_fft_N8)
A_conDF_con_W̄_fft = LinearOperator.(ParameterSpace() × CosFourier(2N,π/di)^2,pfourier, cheb2grid(A_con_cheb,N8_fft).*coefficients.(DF_con_W̄_fft))


I_minus_A_conDF_con_W̄_cheb,I_minus_A_conDF_con_W̄_cheb0 = grid2chebm(_subtract_from_identity(A_conDF_con_W̄_fft),ParameterSpace() × CosFourier(2N,π/di)^2,pfourier,N8)

function _Z0_opnorm(I_minus_A_conDF_con_W̄_cheb0,X₁)
    summ = opnorm(I_minus_A_conDF_con_W̄_cheb0[1],X₁)
    l = length(I_minus_A_conDF_con_W̄_cheb0)
    for k = 1:l 
        summ += exact(2)*opnorm(I_minus_A_conDF_con_W̄_cheb0[k],X₁)
    end
    return summ 
end

Z₁ = _Z0_opnorm(I_minus_A_conDF_con_W̄_cheb0,X₁) + interval(2)*ℒ_∞*(norm_cheb(Vₚ₁ᴺ_cheb0,ℓ¹_τ) + norm_cheb(Vₚ₂ᴺ_cheb0,ℓ¹_τ)) + Z_∞
@show Z₁ 

vals = CAP(sup(Y₀),sup(Z₁),sup(Z₂),sup(R))

# Uncomment this if proving a later segment to verify the continuity.
#=
Wᵢ_end = evaluate.(W̄_cheb_prev,-1)
N_prev = div(length(Wᵢ_end) -1,2) - 1
Wᵢ_end = Sequence(ParameterSpace() × CosFourier(N_prev,π/di)^2, Wᵢ_end)
Wᵢ₊₁_start = Sequence(pfourier0, evaluate.(W̄_cheb,1))
@show norm(Wᵢ_end - Wᵢ₊₁_start,X₁) + R_prev <= vals[2]=#