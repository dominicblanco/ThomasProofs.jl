#Computer assisted proof of a branch of periodic solutions for the 1D Thomas model 
# The following code computes the solution and rigorously proves the results given in section 3.3 of
# "Proving the existence of localized patterns, periodic solutions, and branches of periodic solutions in the 1D Thomas model"  Dominic Blanco

# We provide the data for the approximate solution. 
# From this we can check if the proof of the solution is verified or not.

#####################################################################################################################################################################

# Needed packages
using RadiiPolynomial, LinearAlgebra, JLD2

# Checks the conditions of the Radii-Polynomial Theorem (see Section 3).
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

#################################################### Main Code ####################################################
# Prove Theorem 3.7
N₀ = 27
N = 12
# Defining the parameters
ν = 0.42^2 ; νi = interval(ν)
ν₄ = 21 ; ν₄i = interval(ν₄)
ν₃ = 0.28 ; ν₃i = interval(ν₃)
ν₁ = 8 ; ν₁i = interval(ν₁)
ν₂ = 1 ; ν₂i = interval(ν₂)
ν₅ = 67.46981860371494 ; ν₅i = interval(ν₅)
d = 5 ; di = interval(d)
τ = interval(1.01)
R = interval(5e-9)
fourier0 = CosFourier(N₀,π/di)
fourier = CosFourier(N,π/di)
Ū = load("Ubar_periodic_Th_3_7","U")

#= Prove Theorem 3.8
N₀ = 30
N = 17
# Defining the parameters
ν = 1.08^2 ; νi = interval(ν)
ν₄ = 39.1 ; ν₄i = interval(ν₄)
ν₃ = 0.28 ; ν₃i = interval(ν₃)
ν₁ = 8 ; ν₁i = interval(ν₁)
ν₂ = 1 ; ν₂i = interval(ν₂)
ν₅ = 149.7672 ; ν₅i = interval(ν₅)
d = 50/15 ; di = interval(d)
τ = interval(1.02)
R = interval(5e-8)
fourier0 = CosFourier(N₀,π/di)
fourier = CosFourier(N,π/di)
Ū = load("Ubar_periodic_Th_3_8","U")=#

#= Prove Theorem 3.9
N₀ = 100
N = 100
# Defining the parameters
ν = 0.42^2 ; νi = interval(ν)
ν₄ = 21 ; ν₄i = interval(ν₄)
ν₃ = 0.28 ; ν₃i = interval(ν₃)
ν₁ = 8 ; ν₁i = interval(ν₁)
ν₂ = 1 ; ν₂i = interval(ν₂)
ν₅ = 65 ; ν₅i = interval(ν₅)
d = 10 ; di = interval(d)
τ = interval(1.02)
R = interval(3e-7)
fourier0 = CosFourier(N₀,π/di)
fourier = CosFourier(N,π/di)
Ū = load("Ubar_periodic_Th_3_9","U")=#

Ū_interval = Sequence(fourier0^2, interval.(coefficients(Ū)))
Ū₁_interval = component(Ū_interval,1)
Ū₂_interval = component(Ū_interval,2)

ℓ¹_τ = Ell1(GeometricWeight(τ))
𝒳_τ = NormedCartesianSpace((ℓ¹_τ,ℓ¹_τ),ℓ¹())
# Building the Linear Operator Lₚ defined in Section 5.
@info "Building the Linear Operator"
Lₚ₁₁ = project(Derivative(2), fourier, fourier,Interval{Float64})*νi - UniformScaling(interval(1))
Lₚ₂₁ = LinearOperator(fourier, fourier, Diagonal(interval.(ones(N+1))*(ν₃i*νi - exact(1))))
Lₚ₂₂ = project(Derivative(2), fourier, fourier,Interval{Float64}) - ν₃i*UniformScaling(interval(1))

################## Computing the nonlinear terms ######################################################
@info "Computing the nonlinear terms"
Φ̄ₚ = exact(1) + Ū₁_interval + ν₂i*Ū₁_interval^2 
Φ̄ₚ_inv = inv(Φ̄ₚ)
Ψ̄ₚ = -(ν₁i*νi*Ū₁_interval^2 - ν₁i*Ū₁_interval*Ū₂_interval)
Ψ̄ₚ₁ = -ν₁i*(-Ū₂_interval + exact(2)*νi*Ū₁_interval + νi*Ū₁_interval^2 + ν₂i*Ū₁_interval^2*Ū₂_interval)
Ψ̄ₚ₂ = ν₁i*Ū₁_interval
# Building gₚ, Vₚ₁, and Vₚ₂. Note that these are not exact as we cannot represent the full inverse. We name them for demonstrational purposes.
gₚ = ν₄i + Ψ̄ₚ*Φ̄ₚ_inv 
Vₚ₁_interval = Ψ̄ₚ₁*Φ̄ₚ_inv^2
Vₚ₂_interval = Ψ̄ₚ₂*Φ̄ₚ_inv

#Computation of Aₚ defined in Section 5.1.
@info "Building the Aₚ operator"
DFₚ = LinearOperator(fourier^2, fourier^2, interval.(zeros(2*(N+1),2*(N+1))))
project!(component(DFₚ, 1, 1), Lₚ₁₁ + project(Multiplication(Vₚ₁_interval) ,fourier, fourier, Interval{Float64}))
project!(component(DFₚ, 1, 2), project(Multiplication(Vₚ₂_interval) ,fourier, fourier, Interval{Float64}))
project!(component(DFₚ, 2, 1), Lₚ₂₁)
project!(component(DFₚ, 2, 2), Lₚ₂₂)

Aₚ = interval.(inv(mid.(DFₚ)))
print("Computing norm of Aₚ")
norm_Aₚ = opnorm(Aₚ,𝒳_τ)
@show norm_Aₚ
ℒ_∞ = exact(1)/abs(νi*(interval(N+1)*π/di)^2 + exact(1)) + abs((ν₃i*νi-exact(1))/((νi*(interval(N+1)*π/di)^2 + exact(1))*((interval(N+1)*π/di)^2 + ν₃i))) + exact(1)/abs((interval(N+1)*π/di)^2 + ν₃i)
################## Y₀ BOUND ######################################################
@info "Computing Y₀"
# These are the components of Y₀ expanded. 
# That is, the components of ||Aₚᴺ(LₚŪ + Gₚ(Ū))||_𝒳_τ.
Fₚ = Sequence(fourier^2, [coefficients(project(Derivative(2)*Ū₁_interval*νi - Ū₁_interval + gₚ,fourier)) ; coefficients(project(Derivative(2)*Ū₂_interval - ν₃i*Ū₂_interval + (ν₃i*νi - exact(1))*Ū₁_interval - ν₃i*ν₅i + ν₄i,fourier))])
Y₀¹∞ = Derivative(2)*(Ū₁_interval-project(Ū₁_interval,fourier))*νi - (Ū₁_interval - project(Ū₁_interval,fourier)) + gₚ - project(gₚ,fourier)
Y₀²∞ =  Derivative(2)*(Ū₂_interval-project(Ū₂_interval,fourier)) - ν₃i*(Ū₂_interval-project(Ū₂_interval,fourier)) + (ν₃i*νi - exact(1))*(Ū₁_interval-project(Ū₁_interval,fourier))

Y₀₁ = norm(Aₚ*Fₚ,𝒳_τ) + ℒ_∞*(norm(Y₀¹∞,ℓ¹_τ) + norm(Y₀²∞,ℓ¹_τ)) 
num_term = Ψ̄ₚ*Φ̄ₚ_inv*(exact(1)-Φ̄ₚ*Φ̄ₚ_inv)
Anum_term = Aₚ*Sequence(CosFourier(8N₀,π/di)^2, [coefficients(num_term) ; interval.(zeros(8N₀+1))])
Y₀₂ = (norm(Anum_term,𝒳_τ) + ℒ_∞*norm(Ψ̄ₚ*Φ̄ₚ_inv*(exact(1)-Φ̄ₚ*Φ̄ₚ_inv),ℓ¹_τ))/(exact(1) - norm(exact(1)-Φ̄ₚ*Φ̄ₚ_inv,ℓ¹_τ))
Y₀ = Y₀₁ + Y₀₂
@show Y₀
################## 𝒵₂ BOUND ######################################################
@info "Computing Z₂"    
Z₂₁ = norm(ν₂i^2*Ū₁_interval^3*Ū₂_interval + νi*ν₂i*Ū₁_interval^3 + exact(3)*νi*ν₂i*Ū₁_interval^2 - exact(3)*ν₂i*Ū₁_interval*Ū₂_interval-νi-Ū₂_interval,ℓ¹_τ) + (norm(-exact(1)-exact(3)*ν₂i*Ū₁_interval + ν₂i*Ū₁_interval^3,ℓ¹_τ) + norm(exact(6)*νi*ν₂i*Ū₁_interval + exact(3)*νi*ν₂i*Ū₁_interval^2 - exact(3)*ν₂i*Ū₂_interval + exact(3)*ν₂i^2*Ū₁_interval^2*Ū₂_interval,ℓ¹_τ))*R + (norm(-exact(3)*ν₂i + exact(3)*ν₂i^2*Ū₁_interval^2,ℓ¹_τ) + norm(exact(3)*νi*ν₂i + exact(3)*νi*ν₂i*Ū₁_interval + exact(3)*ν₂i*Ū₁_interval,ℓ¹_τ))*R^2 + (norm(exact(3)*ν₂i^2*Ū₁_interval,ℓ¹_τ) + norm(νi*ν₂i + ν₂i*Ū₂_interval,ℓ¹_τ))*R^3 + ν₂i^2*R^4
Z₂₂ = norm(Φ̄ₚ_inv,ℓ¹_τ)^3/(exact(1) - norm(exact(1)-Φ̄ₚ*Φ̄ₚ_inv,ℓ¹_τ) - R*norm(Φ̄ₚ_inv,ℓ¹_τ))^3
Z₂₃ = (norm(exact(1)-ν₂i*Ū₁_interval^2,ℓ¹_τ) + exact(2)*ν₂i*norm(Ū₁_interval,ℓ¹_τ)*R + ν₂i*R^2)*norm(Φ̄ₚ_inv,ℓ¹_τ)^2/(exact(1) - norm(exact(1)-Φ̄ₚ*Φ̄ₚ_inv,ℓ¹_τ) - R*norm(Φ̄ₚ_inv,ℓ¹_τ))^2
Z₂ = exact(2)*ν₁i*(norm_Aₚ + ℒ_∞)*(Z₂₁*Z₂₂ + Z₂₃)
@show Z₂
################## 𝒵₁ BOUND ###################################################### 
#These are the actual Vₚⱼᴺ for j = 1,2.
Vₚ₁ᴺ_interval = project(Vₚ₁_interval,fourier)
Vₚ₂ᴺ_interval = project(Vₚ₂_interval,fourier)

Z_∞₁ = norm(Vₚ₁_interval - Vₚ₁ᴺ_interval,ℓ¹_τ) + norm(Ψ̄ₚ₁*Φ̄ₚ_inv^2*(exact(1)-Φ̄ₚ^2*Φ̄ₚ_inv^2),ℓ¹_τ)/(exact(1) - norm(exact(1)-Φ̄ₚ^2*Φ̄ₚ_inv^2,ℓ¹_τ))
Z_∞₂ = norm(Vₚ₂_interval - Vₚ₂ᴺ_interval,ℓ¹_τ) + norm(Ψ̄ₚ₂*Φ̄ₚ_inv*(exact(1)-Φ̄ₚ*Φ̄ₚ_inv),ℓ¹_τ)/(exact(1) - norm(exact(1)-Φ̄ₚ*Φ̄ₚ_inv,ℓ¹_τ))
Z_∞ = (norm_Aₚ + ℒ_∞)*(Z_∞₁ + Z_∞₂)

𝕍ₚ₁_2N = project(Multiplication(Vₚ₁ᴺ_interval),CosFourier(2N,π/di),fourier,Interval{Float64})
𝕍ₚ₂_2N = project(Multiplication(Vₚ₂ᴺ_interval),CosFourier(2N,π/di),fourier,Interval{Float64})
DFₚ_2N = LinearOperator(CosFourier(2N,π/di)^2,fourier^2, interval.(zeros(2*(N+1),2*(2N+1))))
project!(component(DFₚ_2N,1,1), Lₚ₁₁ + 𝕍ₚ₁_2N)
project!(component(DFₚ_2N,1,2), 𝕍ₚ₂_2N)
project!(component(DFₚ_2N,2,1), Lₚ₂₁)
project!(component(DFₚ_2N,2,2), Lₚ₂₂)

Z₁ = opnorm(UniformScaling(interval(1)) - Aₚ*DFₚ_2N,𝒳_τ) + exact(2)*ℒ_∞*(norm(Vₚ₁ᴺ_interval,ℓ¹_τ) + norm(Vₚ₂ᴺ_interval,ℓ¹_τ)) + Z_∞
@show Z₁ 

vals = CAP(sup(Y₀),sup(Z₁),sup(Z₂),sup(R))