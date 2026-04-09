#Computer assisted proof of a branch of periodic solutions for the 1D Thomas model 
# The following code computes the solution and rigorously proves the results given in section 3.3 of
# "Proving the existence of localized patterns, periodic solutions, and branches of periodic solutions in the 1D Thomas model"  Dominic Blanco

# We provide the data for the approximate solution. 
# From this we can check if the proof of the solution is verified or not.

#####################################################################################################################################################################

# Needed packages
using RadiiPolynomial, LinearAlgebra, JLD2

#Builds Matrix P for trace
function trace2(N)
    P = Int.(zeros(N+1,N+1))
    P[2:end,2:end] = Matrix(I,N,N)
    for n = 1:N
        P[1,n+1] = (Int(-2*(-1)^n))
    end
    return P
end

function φ(A,B,C,D)
    O₁ = max(A,D) + max(B,C)
    O₂ = sqrt(A^2 + D^2 + B^2 + C^2)
    return min(O₁,O₂)
end

function _conv_small(u,v,N)
    #Computes u*v only up to order N
    order_u = order(space(u))[1]
    order_v = order(space(v))[1]
    C = Sequence(CosFourier(N,frequency(u)[1]), (interval.(zeros(N+1))))
    for i ∈ 0:N
        Cᵢ = interval(zero(Float64))
        for j ∈ max(i-order_u, -order_v):min(i+order_u, order_v)
            tu = abs(i-j)
            tv = abs(j)
            Cᵢ += u[tu] * v[tv]
        end
        C[i] = Cᵢ
    end
    return C
end

# Computes the roots of the polynomial ax² + bx + c
function quad_roots(a, b, c)
    D = b^2 - interval(4)*a*c
    if sup(D) < 0
        D = Complex(D)
        sqrtD = sqrt(D)
        return [(-b + sqrtD) / (interval(2)*a), (-b - sqrtD) / (interval(2)*a)]
    else
        sqrtD = sqrt(D)
        return [(-b + sqrtD) / (interval(2)*a), (-b - sqrtD) / (interval(2)*a)]
    end
end

# Computes the roots of a cubic.
function _cardano_roots(a, b, c, d)
    # 1. Normalize coefficients
    a, b, c, d = ComplexF64.((a, b, c, d))
    
    # 2. Parameters for depressed cubic y^3 + py + q = 0
    p = (3a*c - b^2) / (3a^2)
    q = (2b^3 - 9a*b*c + 27a^2*d) / (27a^3)
    
    # 3. Discriminant and Cardano components
    Δ = (q/2)^2 + (p/3)^3
    sqrt_Δ = sqrt(Δ)
    
    # Calculate u; handle u=0 to avoid division errors
    u_cubed = -q/2 + sqrt_Δ
    u = u_cubed^(1/3)
    
    # Define the three cube roots of unity
    ω = -0.5 + (sqrt(3)/2)im
    roots_u = [u, u*ω, u*ω^2]
    
    # 4. Calculate roots using the constraint v = -p / (3u)
    # If u is 0, then v^3 = -q
    roots = map(roots_u) do ui
        if abs(ui) > 1e-15
            vi = -p / (3ui)
            return ui + vi - b/(3a)
        else
            # Special case when u is zero
            return (-q)^(1/3) - b/(3a)
        end
    end
    l = length(roots) 
    for k = 1:l 
        if abs(imag(roots[k])) < 1e-14 
            return real(roots[k])
        else
            continue 
        end
    end
end

function integral_computation(ν,ν₃,λ₅,λ₆,λ₇)
    # This function computes the integral squared of $\frac{(a₁ξ² + a₂)²}{l_{den}(ξ)^2}$

    # Compute roots of l_den
    y₁², y₂² = Complex.(quad_roots(interval(2π)^4*ν, interval(2π)^2*(ν*ν₃ - λ₆ + interval(1)), -ν₃*(λ₆ - interval(1)) - λ₅*λ₇))
    y₁ = sqrt(Complex(y₁²))
    y₂ = sqrt(Complex(y₂²))

    if (imag(y₁)) < 0
        y₁ = -y₁
    elseif (imag(y₂)) < 0
        y₂ = -y₂
    end

    residue_y₁(a1, a2) = interval(2)*(a1*y₁^2 + a2)*(interval(4)*a1*y₁^2*(y₁^2 - y₂^2) - (a1*y₁^2 + a2)*(interval(5)*y₁^2 - y₂^2)) / (interval(2^3) * interval(2π)^8 * ν^2 * y₁^3*(y₁^2 - y₂^2)^3)
    residue_y₂(a1, a2) = interval(2)*(a1*y₂^2 + a2)*(interval(4)*a1*y₂^2*(y₂^2 - y₁^2) - (a1*y₂^2 + a2)*(interval(5)*y₂^2 - y₁^2)) / (interval(2^3) * interval(2π)^8 * ν^2 * y₂^3*(y₂^2 - y₁^2)*(y₂^2 - y₁^2)*(y₂^2 - y₁^2))
    
    # Compute the integrals
    resl11_1 = residue_y₁(-interval(2π)^2*ν, λ₆ - interval(1))
    resl11_2 = residue_y₂(-interval(2π)^2*ν, λ₆ - interval(1))

    int_l11_squared = interval(2π*1im)*(resl11_1 + resl11_2)
    if inf(abs(imag(int_l11_squared))) == 0
        int_l11_squared = real(int_l11_squared)
    else
        error("Integral l11 has significant imaginary part")
    end

    resl12_1 = residue_y₁(interval(0), λ₇)
    resl12_2 = residue_y₂(interval(0), λ₇)

    int_l12_squared = interval(2π*1im)*(resl12_1 + resl12_2)
    if inf(abs(imag(int_l12_squared))) == 0
        int_l12_squared = real(int_l12_squared)
    else
        error("Integral l12 has significant imaginary part")
    end

    resl21_1 = residue_y₁(interval(0), λ₅)
    resl21_2 = residue_y₂(interval(0), λ₅)

    int_l21_squared = interval(2π*1im)*(resl21_1 + resl21_2)
    if inf(abs(imag(int_l21_squared))) == 0
        int_l21_squared = real(int_l21_squared)
    else
        error("Integral l21 has significant imaginary part")
    end

    resl22_1 = residue_y₁(-interval(2π)^2, -ν₃)
    resl22_2 = residue_y₂(-interval(2π)^2, -ν₃)

    int_l22_squared = interval(2π*1im)*(resl22_1 + resl22_2)
    if inf(abs(imag(int_l22_squared))) == 0
        int_l22_squared = real(int_l22_squared)
    else
        error("Integral l22 has significant imaginary part")
    end

    return int_l11_squared, int_l12_squared, int_l21_squared, int_l22_squared
end

# This computes generally one of the four terms needed to compute ||𝓁⁻¹||_ℳ₁
function _compute_ℳ₁_component(a₁,a₂,ν,ν₃,λ₅,λ₆,λ₇,f_num,f_denom)
    f(ξ) = f_num(ξ)/f_denom(ξ)
    b₁ = ν
    b₂ = ν*ν₃ - λ₆ + interval(1)
    b₃ = -ν₃*(λ₆ - interval(1)) - λ₅*λ₇
    r₁,r₂ = quad_roots(interval(16)*interval(π^4)*a₁*b₁,interval(8)*interval(π^2)*a₂*b₁,a₂*b₂ - a₁*b₃)
    max_val = abs(f(interval(0)))
    if inf(r₁) > 0
        max_val = max(max_val,abs(f(sqrt(r₁))),abs(f(-sqrt(r₁))))
    end
    if inf(r₂) > 0
        max_val = max(max_val,abs(f(sqrt(r₂))),abs(f(-sqrt(r₂))))
    end
    return max_val
end

function _compute_qijs(U₁,U₂,ν₁,ν₂,λ₁,λ₃,λ₄)
    μ₁ = ν₁*ν₂
    μ₂ = λ₃ + exact(2)*λ₁*λ₃*ν₂ - λ₄*ν₂
    μ₃ = exact(2)*λ₁*ν₁*ν₂ 
    μ₄ = exact(2)*λ₁*λ₃ + exact(2)*λ₃ + exact(2)*λ₁^2*λ₃*ν₂
    μ₅ = λ₁^2*ν₁*ν₂ - ν₁
    μ₆ = λ₁*λ₄+λ₄+λ₁^2*λ₄*ν₂

    #We change the naming convention for easier implementation.
    lambda_1 = λ₁ 
    lambda_3 = λ₃ 
    lambda_4 = λ₄ 
    nu_1 = ν₁ 
    nu_2 = ν₂
    mu_1 = ν₁*ν₂
    mu_2 = λ₃ + exact(2)*λ₁*λ₃*ν₂ - λ₄*ν₂
    mu_3 = exact(2)*λ₁*ν₁*ν₂ 
    mu_4 = exact(2)*λ₁*λ₃ + exact(2)*λ₃ + exact(2)*λ₁^2*λ₃*ν₂
    mu_5 = λ₁^2*ν₁*ν₂ - ν₁
    mu_6 = λ₁*λ₄+λ₄+λ₁^2*λ₄*ν₂
   
    υ⁰¹₀₀ = μ₅ + exact(2)*λ₁*μ₅ + λ₁^2*μ₅ + exact(2)*λ₁^2*ν₂*μ₅ + exact(2)*λ₁^3*ν₂*μ₅ + λ₁^4*ν₂^2*μ₅ 
    υ⁰¹₁₀ = μ₃ + exact(2)*λ₁*μ₃ + λ₁^2*μ₃ + exact(2)*λ₁^2*ν₂*μ₃ + exact(2)*λ₁^3*ν₂*μ₃ + λ₁^4*ν₂^2*μ₃ + exact(2)*μ₅+exact(2)*λ₁*μ₅+exact(4)*λ₁*ν₂*μ₅ + exact(6)*λ₁^2*ν₂*μ₅ + exact(4)*λ₁^3*ν₂^2*μ₅
    υ⁰¹₂₀ = μ₁ + exact(2)*λ₁*μ₁ + λ₁^2*μ₁ + exact(2)*λ₁^2*ν₂*μ₁ + exact(2)*λ₁^3*ν₂*μ₁ + exact(2)*μ₃ + exact(2)*λ₁*μ₃ + exact(4)*λ₁*ν₂*μ₃ + exact(6)*λ₁^2*ν₂*μ₃ + exact(4)*λ₁^3*ν₂^2*μ₃ + μ₅ + exact(2)*ν₂*μ₅ + exact(6)*λ₁*ν₂*μ₅ + exact(6)*λ₁^2*ν₂^2*μ₅ 
    υ⁰¹₃₀ = exact(2)*μ₁ + exact(2)*λ₁*μ₁ + exact(4)*λ₁*ν₂*μ₁ + exact(6)*λ₁^2*ν₂*μ₁ + exact(4)*λ₁^3*ν₂^2*μ₁ + μ₃ + exact(2)*ν₂*μ₃ + exact(6)*λ₁*ν₂*μ₃ + exact(6)*λ₁^2*ν₂^2*μ₃ + exact(2)*ν₂*μ₅ + exact(4)*λ₁*ν₂^2*μ₅
    υ⁰¹₄₀ = mu_1 + exact(2)*nu_2*mu_1 + exact(6)*lambda_1*nu_2*mu_1 + exact(6)*lambda_1^2*nu_2^2*mu_1 + exact(2)*nu_2*mu_3 + exact(4)*lambda_1*nu_2^2*mu_3 + nu_2^2*mu_5
    υ⁰¹₅₀ = exact(2)*nu_2*mu_1 + exact(4)*lambda_1*nu_2^2*mu_1 + nu_2^2*mu_3
    υ⁰¹₆₀ = nu_2^2*mu_1 

    υ¹¹₀₀ = mu_3 + exact(2)*lambda_1*mu_3 + lambda_1^2*mu_3 + exact(2)*lambda_1^2*nu_2*mu_3 + exact(2)*lambda_1^2*nu_2*mu_3 + exact(2)*lambda_1^3*nu_2*mu_3 + lambda_1^4*nu_2^2*mu_3
    υ¹¹₁₀ = exact(2)*mu_1 + exact(4)*lambda_1*mu_1 + exact(2)*lambda_1^2*mu_1 + exact(4)*lambda_1^2*nu_2*mu_1 + exact(4)*lambda_1^2*nu_2*mu_1 + exact(4)*lambda_1^3*nu_2*mu_1 + exact(2)*lambda_1^4*nu_2^2*mu_1 + exact(2)*mu_3 + exact(2)*lambda_1*mu_3 + exact(4)*lambda_1*nu_2*mu_3 
    υ¹¹₂₀ = exact(4)*mu_1 + exact(4)*lambda_1 *mu_1 + exact(8)*lambda_1 *nu_2 *mu_1 + exact(12)*lambda_1^2 *nu_2 *mu_1 + exact(8)*lambda_1^3 *nu_2^2 *mu_1 + mu_3 + exact(2)*nu_2 *mu_3 + exact(6)*lambda_1 *nu_2 *mu_3 + exact(6)*lambda_1^2 *nu_2^2 *mu_3 
    υ¹¹₃₀ = exact(2)*mu_1 + exact(4)*nu_2 *mu_1 + exact(12)*lambda_1* nu_2* mu_1 + exact(12)*lambda_1^2 *nu_2^2 *mu_1 + exact(2)*nu_2 *mu_3 + exact(4)*lambda_1 *nu_2^2 *mu_3 
    υ¹¹₄₀ = exact(4)*nu_2* mu_1 + exact(8)*lambda_1 *nu_2^2 *mu_1 + nu_2^2 *mu_3 
    υ¹¹₅₀ = exact(2)*nu_2^2 * mu_1

    υ¹⁰₀₀ = mu_4 + exact(2)*lambda_1 *mu_4 + lambda_1^2* mu_4 + exact(2)*lambda_1^2 *nu_2* mu_4 + exact(2)*lambda_1^3 *nu_2 *mu_4 + lambda_1^4 *nu_2^2 *mu_4 - exact(2)*mu_6 - exact(2)*lambda_1 *mu_6 - exact(4)*lambda_1 *nu_2 *mu_6 - exact(6)*lambda_1^2 *nu_2 *mu_6 - exact(4)*lambda_1^3 *nu_2^2* mu_6
    υ¹⁰₀₁ = exact(2)*lambda_1 *mu_3 + lambda_1^2*mu_3 + exact(2)*lambda_1^2 *nu_2* mu_3 + exact(2)*lambda_1^3 *nu_2 *mu_3 + lambda_1^4 *nu_2^2 *mu_3 - exact(2)*mu_5 - exact(2)*lambda_1 *mu_5 - exact(4)*lambda_1 *nu_2 *mu_5 - exact(6)*lambda_1^2 *nu_2 *mu_5 - exact(4)*lambda_1^3 *nu_2^2 *mu_5
    υ¹⁰₁₀ = exact(2)*mu_2 + exact(4)*lambda_1* mu_2 + exact(2)*lambda_1^2 *mu_2 + exact(4)*lambda_1^2 *nu_2* mu_2 + exact(4)*lambda_1^3 *nu_2* mu_2 + exact(2)*lambda_1^4 *nu_2^2 *mu_2 - exact(2)*mu_6 - exact(4)*nu_2 *mu_6 - exact(12)*lambda_1 *nu_2 *mu_6 - exact(12)*lambda_1^2 *nu_2^2 *mu_6 
    υ¹⁰₁₁ = exact(2)*mu_1 + exact(4)*lambda_1* mu_1 + exact(2)*lambda_1^2 *mu_1 + exact(4)*lambda_1^2 *nu_2 *mu_1 + exact(4)*lambda_1^3 *nu_2 *mu_1 + exact(2)*lambda_1^4 *nu_2^2 *mu_1 - exact(2)*mu_5 - exact(4)*nu_2 *mu_5 - exact(12)*lambda_1 *nu_2 *mu_5 - exact(12)*lambda_1^2 *nu_2^2 *mu_5 
    υ¹⁰₂₀ = exact(2)*mu_2 + exact(2)*lambda_1 *mu_2 + exact(4)*lambda_1 *nu_2 *mu_2 + exact(6)*lambda_1^2 *nu_2 *mu_2 + exact(4)*lambda_1^3 *nu_2^2 *mu_2 - mu_4 - exact(2)*nu_2 *mu_4 - exact(6)*lambda_1 *nu_2 *mu_4 - exact(6)*lambda_1^2 *nu_2^2 *mu_4 - exact(6)*nu_2 *mu_6 - exact(12)*lambda_1 *nu_2^2 *mu_6 
    υ¹⁰₂₁ = exact(2)*mu_1 + exact(2)*lambda_1* mu_1 + exact(4)*lambda_1* nu_2* mu_1 + exact(6)*lambda_1^2 *nu_2 *mu_1 + exact(4)*lambda_1^3 *nu_2^2* mu_1 - mu_3 - exact(2)*nu_2* mu_3 - exact(6)*lambda_1 *nu_2 *mu_3 - exact(6)*lambda_1^2 *nu_2^2 *mu_3 - exact(6)*nu_2 *mu_5 - exact(12)*lambda_1* nu_2^2* mu_5 
    υ¹⁰₃₀ = -exact(4)*nu_2 *mu_4 - exact(8)*lambda_1 *nu_2^2 *mu_4 - exact(4)*nu_2^2*mu_6 
    υ¹⁰₃₁ = -exact(4)*nu_2 *mu_3 - exact(8)*lambda_1 *nu_2^2 *mu_3 - exact(4)*nu_2^2 *mu_5
    υ¹⁰₄₀ =  -exact(2)*nu_2 *mu_2 - exact(4)*lambda_1 *nu_2^2 *mu_2 - exact(3)*nu_2^2* mu_4 
    υ¹⁰₄₁ = -exact(2)*nu_2 *mu_1 - exact(4)*lambda_1 *nu_2^2 *mu_1 - exact(3)*nu_2^2* mu_3
    υ¹⁰₅₀ =  exact(2)*nu_2^2 *mu_2
    υ¹⁰₅₁ =  -exact(2)*nu_2^2*mu_1

    υ²¹₀₀ = mu_1 + exact(2)*lambda_1 *mu_1 + lambda_1^2 *mu_1 + exact(2)*lambda_1^2 *nu_2 *mu_1 + exact(2)*lambda_1^3 *nu_2 *mu_1 + lambda_1^4 *nu_2^2* mu_1
    υ²¹₁₀ = exact(2)*mu_1 + exact(2)*lambda_1* mu_1 + exact(4)*lambda_1 *nu_2 *mu_1 + exact(6)*lambda_1^2 *nu_2 *mu_1 + exact(4)*lambda_1^3*nu_2^2* mu_1 
    υ²¹₂₀ = mu_1 *exact(2)*nu_2 *mu_1 + exact(6)*lambda_1 *nu_2* mu_1 + exact(6)*lambda_1^2 *nu_2^2 *mu_1 
    υ²¹₃₀ = exact(2)*nu_2 *mu_1 + exact(4)*lambda_1 *nu_2^2 *mu_1 
    υ²¹₄₀ = nu_2^2*mu_1

    υ²⁰₀₀ = mu_2 + exact(2)*lambda_1* mu_2 + lambda_1^2* mu_2 + exact(2)*lambda_1^2* nu_2* mu_2 + exact(2)*lambda_1^3* nu_2 *mu_2 + lambda_1^4 *nu_2^2 *mu_2 - mu_6 - exact(2)*nu_2 *mu_6 - exact(6)*lambda_1 *nu_2* mu_6 - exact(6)*lambda_1^2 *nu_2^2 *mu_6
    υ²⁰₀₁ = mu_1 + exact(2)*lambda_1 *mu_1 + lambda_1^2* mu_1 + exact(2)*lambda_1^2* nu_2 *mu_1 + exact(2)*lambda_1^3* nu_2 *mu_1 + lambda_1^4 *nu_2^2*mu_1 - mu_5 - exact(2)*nu_2 *mu_5 - exact(6)*lambda_1 *nu_2 *mu_5 - exact(6)*lambda_1^2 *nu_2^2 *mu_5
    υ²⁰₁₀ = exact(2)*mu_2 + exact(2)lambda_1* mu_2 + exact(4)*lambda_1* nu_2* mu_2 + exact(6)*lambda_1^2* nu_2* mu_2 + exact(4)*lambda_1^3* nu_2^2 *mu_2 -mu_4 - exact(2)*nu_2* mu_5 - exact(6)*lambda_1 *nu_2* mu_4 - exact(6)*lambda_1^2 *nu_2^2 *mu_4 - exact(6)*nu_2 *mu_6 - exact(12)*lambda_1* nu_2^2* mu_6 
    υ²⁰₁₁ = exact(2)*mu_1 + exact(2)*lambda_1* mu_1 + exact(4)*lambda_1* nu_2* mu_1 + exact(6)*lambda_1^2* nu_2* mu_1 + exact(4)*lambda_1^3 *nu_2^2* mu_1 - mu_3 - exact(2)*nu_2* mu_3 - exact(6)*lambda_1 *nu_2 *mu_3 - exact(6)*lambda_1^2 *nu_2^2 *mu_3 - exact(6)*nu_2* mu_5 - exact(12)*lambda_1* nu_2^2* mu_5 
    υ²⁰₂₀ = -exact(6)*nu_2 *mu_4 - exact(12)*lambda_1 *nu_2^2 *mu_4 - exact(6)*nu_2^2* mu_6
    υ²⁰₂₁ =  -exact(6)*nu_2 *mu_3 - exact(12)*lambda_1 *nu_2^2*mu_3 - exact(6)*nu_2^2* mu_5 
    υ²⁰₃₀ = -exact(4)*nu_2* mu_2 -exact(8)*lambda_1 *nu_2^2 *mu_2 - exact(6)*nu_2^2* mu_4 
    υ²⁰₃₁ = -exact(4)*nu_2 *mu_1 - exact(8)*lambda_1* nu_2^2* mu_1 -exact(6)*nu_2^2 *mu_3 
    υ²⁰₄₀ = -exact(5)*nu_2^2*mu_2 
    υ²⁰₄₁ = -exact(5)*nu_2^2*mu_1 

    υ³⁰₀₀ = -exact(2)*nu_2 *mu_6 - exact(4)*lambda_1* nu_2^2 *mu_6 - exact(2)*nu_2 
    υ³⁰₀₁ = -exact(2)*nu_2 *mu_5 - exact(4)*lambda_1 *nu_2^2 *mu_5 
    υ³⁰₁₀ = -exact(2)*nu_2* mu_4 -exact(4)*lambda_1* nu_2^2* mu_4 - exact(4)*nu_2^2* mu_6 
    υ³⁰₁₁ = -exact(2)*nu_2* mu_3 - exact(4)*lambda_1* nu_2^2* mu_3 - exact(4)*nu_2^2* mu_5 
    υ³⁰₂₀ = -exact(2)*nu_2* mu_2 - exact(4)*lambda_1* nu_2^2* mu_2 - exact(4)*nu_2^2* mu_4 
    υ³⁰₂₁ = -exact(2)*nu_2* mu_1 - exact(4)*lambda_1 *nu_2^2* mu_1 - exact(4)*nu_2^2* mu_3 
    υ³⁰₃₀ = -exact(4)*nu_2^2 *mu_2 
    υ³⁰₃₁ = -exact(4)*nu_2^2*mu_1

    υ⁴⁰₀₀ = -nu_2^2*mu_6 
    υ⁴⁰₁₀ = -nu_2^2*mu_4 
    υ⁴⁰₂₀ = -nu_2^2*mu_2 
    υ⁴⁰₀₁ = -nu_2^2*mu_5 
    υ⁴⁰₁₁ = -nu_2^2*mu_3 
    υ⁴⁰₂₁ = -nu_2^2*mu_1

    q₀₁ = υ⁰¹₁₀*U₁ + υ⁰¹₂₀*U₁^2 + υ⁰¹₃₀*U₁^3 + υ⁰¹₄₀*U₁^4 + υ⁰¹₅₀*U₁^5 + υ⁰¹₆₀*U₁^6 
    q₁₁ = υ¹¹₁₀*U₁ + υ¹¹₂₀*U₁^2 + υ¹¹₃₀*U₁^3 + υ¹¹₄₀*U₁^4 + υ¹¹₅₀*U₁^5
    q₁₀ = υ¹⁰₁₀*U₁ + υ¹⁰₀₁*U₂ + υ¹⁰₁₁*U₁*U₂ + υ¹⁰₂₀*U₁^2 + υ¹⁰₂₁*U₁^2*U₂ + υ¹⁰₃₀*U₁^3 +υ¹⁰₃₁*U₁^3*U₂ + υ¹⁰₄₀*U₁^4 + υ¹⁰₄₁*U₁^4*U₂ + υ¹⁰₅₀*U₁^5 + υ¹⁰₅₁*U₁^5*U₂
    q₂₁ = υ²¹₁₀*U₁ + υ²¹₂₀*U₁^2 + υ²¹₃₀*U₁^3 + υ²¹₄₀*U₁^4
    q₂₀ = υ²⁰₁₀*U₁ + υ²⁰₀₁*U₂ + υ²⁰₁₁*U₁*U₂ + υ²⁰₂₀*U₁^2 + υ²⁰₂₁*U₁^2*U₂ + υ²⁰₃₀*U₁^3 +υ²⁰₃₁*U₁^3*U₂ + υ²⁰₄₀*U₁^4 + υ²⁰₄₁*U₁^4*U₂
    q₃₀ = υ³⁰₁₀*U₁ + υ³⁰₀₁*U₂ + υ³⁰₁₁*U₁*U₂ + υ³⁰₂₀*U₁^2 + υ³⁰₂₁*U₁^2*U₂ + υ³⁰₃₀*U₁^3 +υ³⁰₃₁*U₁^3*U₂
    q₄₀ = υ⁴⁰₁₀*U₁ + υ⁴⁰₀₁*U₂ + υ⁴⁰₁₁*U₁*U₂ + υ⁴⁰₂₀*U₁^2 + υ⁴⁰₂₁*U₁^2*U₂
    return υ⁰¹₀₀,υ¹⁰₀₀,υ¹¹₀₀,υ²⁰₀₀,υ²¹₀₀,υ³⁰₀₀,υ⁴⁰₀₀,q₀₁,q₁₁,q₁₀,q₂₁,q₂₀,q₃₀,q₄₀
end

# Computes the constants Cⱼ and a.
function _compute_C_a(νi,ν₃i,λ₅i,λ₆i,λ₇i)
    if sup((νi*ν₃i - λ₆i+interval(1))*(νi*ν₃i - λ₆i+interval(1)) + interval(4)*νi*(ν₃i*λ₆i - ν₃i + λ₅i*λ₇i)) < 0
        y = sqrt( (λ₆i - interval(1) - νi*ν₃i + interval(1im)*sqrt(-(νi*ν₃i - λ₆i+interval(1))*(νi*ν₃i - λ₆i+interval(1)) - interval(4)*νi*(ν₃i*λ₆i - ν₃i + λ₅i*λ₇i))) /(interval(2)*νi))*interval(1)/interval(2π)
        z₁ = interval(2π)*interval(1im)*y
        z₂ = interval(-2π)*interval(1im)*conj(y)
        if (sup(real(z₁)) < 0) & (sup(real(z₂)) < 0)
            z₁ = -z₁
            z₂ = -z₂ 
        end
    elseif isstrictless(sqrt((νi*ν₃i - λ₆i+interval(1))*(νi*ν₃i - λ₆i+interval(1)) + interval(4)*νi*(ν₃i*λ₆i - ν₃i + λ₅i*λ₇i)), νi*ν₃i - λ₆i+interval(1))
        z₁ = sqrt( (νi*ν₃i - λ₆i+interval(1) + sqrt((νi*ν₃i - λ₆i+interval(1))*(νi*ν₃i - λ₆i+interval(1)) + interval(4)*νi*(ν₃i*λ₆i - ν₃i + λ₅i*λ₇i))) / (interval(2)*νi))
        z₂ = sqrt( (νi*ν₃i - λ₆i+interval(1) - sqrt((νi*ν₃i - λ₆i+interval(1))*(νi*ν₃i - λ₆i+interval(1)) + interval(4)*νi*(ν₃i*λ₆i - ν₃i + λ₅i*λ₇i))) / (interval(2)*νi))
    else
        error(`Assumption 1 is not satisfied`)
    end

    _Cj(d₁, d₂, νi) = (abs(d₁*z₂) + abs(d₂ / z₂)) / abs(interval(2) * νi * (z₁*z₁ - z₂*z₂))
    C1 = _Cj(interval(-1), -ν₃i, νi)
    C2 = _Cj(interval(0), λ₇i, νi)
    C3 = _Cj(interval(0), λ₅i, νi)
    C4 = _Cj(-νi, λ₆i-interval(1), νi)
    a = min(real(z₁), real(z₂))

    return a,C1,C2,C3,C4
end

# Checks the conditions of the Radii-Polynomial Theorem (see Section 2.3.1).
function CAP(𝒴₀,𝒵₁,𝒵₂)
    if 𝒵₁ > 1
        display("Z₁ is too big")
        return 𝒵₁
    elseif 2𝒴₀*𝒵₂ > (1-𝒵₁)^2
        display("The condition 2𝒴₀*𝒵₂ < (1-𝒵₁)² is not satisfied")
        return 𝒴₀,𝒵₁,𝒵₂
    else
        display("The computer assisted proof was successful!")
        return 𝒴₀,𝒵₁,𝒵₂
    end
end


############################################### Main Code ###############################################
# Prove Theorem 2.14
Ū = load("Ubar_localized_Th_2_14","U")
N₀ = 500
N = 300
# Defining the parameters
ν = 1.08^2 ; νi = interval(ν)
ν₄ = 39.1
ν₃ = 0.28 ; ν₃i = interval(ν₃)
ν₅ = 150
ν₁ = 8 ; ν₁i = interval(ν₁)
ν₂ = 1 ; ν₂i = interval(ν₂)
r₀ = interval(3e-10)
d = 30

#= Prove Theorem 2.15
Ū = load("Ubar_localized_Th_2_15","U")
N₀ = 750
N = 300
# Defining the parameters
ν = 1.08^2 ; νi = interval(ν)
ν₃ = 0.28 ; ν₃i = interval(ν₃)
ν₁ = 8 ; ν₁i = interval(ν₁)
ν₂ = 1 ; ν₂i = interval(ν₂)
ν₄ = 39.10658 
ν₅ = 150
r₀ = interval(2e-9)
d = 50=#

#= Prove Theorem 2.16 
Ū = load("Ubar_localized_Th_2_16","U")
N₀ = 750
N = 300
# Defining the parameters
ν = 1.08^2 ; νi = interval(ν)
ν₃ = 0.28 ; ν₃i = interval(ν₃)
ν₁ = 8 ; ν₁i = interval(ν₁)
ν₂ = 1 ; ν₂i = interval(ν₂)
ν₄ = 39.10658
ν₅ = 149.83  
r₀ = interval(4e-9) 
d = 50=#

#= Prove Theorem 2.17
Ū = load("Ubar_localized_Th_2_17","U")
N₀ = 1800
N = 650
# Defining the parameters
ν = 0.46^2 ; νi = interval(ν)
ν₄ = 21.3
ν₃ = 0.28 ; ν₃i = interval(ν₃)
ν₅ = 64.5
ν₁ = 8 ; ν₁i = interval(ν₁)
ν₂ = 1 ; ν₂i = interval(ν₂)
r₀ = interval(4e-11)
d = 57=#

#= Prove Theorem 2.18
Ū = load("Ubar_localized_Th_2_18","U")
N₀ = 2100
N = 2100
# Defining the parameters
ν = 0.47733^2 ; νi = interval(ν)
ν₄ = 21
ν₃ = 0.2799999 ; ν₃i = interval(ν₃)
ν₅ = 65.04
ν₁ = 8 ; ν₁i = interval(ν₁)
ν₂ = 0.9498 ; ν₂i = interval(ν₂)
r₀ = interval(4e-11)
d = 62=#

λ₁ = _cardano_roots(ν₃*ν₂, -((ν₄*ν₂-1)*ν₃-ν₁), (ν₅*ν₁-(ν₄-1))*ν₃-ν₄*ν₁, -ν₄*ν₃) ; λ₁i = interval(λ₁)
λ₂ = (1 + λ₁+ν₂*λ₁^2)*(ν₄-λ₁)/(ν₁*λ₁) ; λ₂i = interval(λ₂)
λ₃ = ν₁*ν + (λ₁ - ν₄)*ν₂ ; λ₃i = interval(λ₃)
λ₄ = ν₁*λ₂ + (1+ 2ν₂*λ₁)*(λ₁ - ν₄) + ν₁*λ₁*ν ; λ₄i = interval(λ₄)
λ₅ = ν₃*ν - 1 ; λ₅i = interval(λ₅)
λ₆ = -(λ₁*λ₄ + λ₄ + λ₁^2*λ₄*ν₂)/(1 + λ₁ + ν₂*λ₁^2)^2 ; λ₆i = interval(λ₆) 
λ₇ = ν₁*λ₁/(1 + λ₁ + ν₂*λ₁^2) ; λ₇i = interval(λ₇)
di = interval(d)
fourier0 = CosFourier(N₀,π/di)
fourier = CosFourier(N,π/di)

# Computing the trace.
setprecision(128)
𝒫 = LinearOperator(fourier0,fourier0,interval.(big.(trace2(N₀))))
Ū₁_big = Sequence(fourier0, coefficients(𝒫)*interval.(big.(coefficients(component(Ū,1)))))
Ū₂_big = Sequence(fourier0, coefficients(𝒫)*interval.(big.(coefficients(component(Ū,2)))))
Ū₁_interval = interval.(Float64.(inf.(Ū₁_big ),RoundDown),Float64.(sup.(Ū₁_big ),RoundUp) )
Ū₂_interval = interval.(Float64.(inf.(Ū₂_big ),RoundDown),Float64.(sup.(Ū₂_big ),RoundUp) )

Ū_interval = Sequence(fourier0^2, [Ū₁_interval[:] ; Ū₂_interval[:]])

# Building the Linear Operator L defined.
@info "Building the Linear Operator"
L₁₁ = diag(coefficients(project(Derivative(2), fourier, fourier,Interval{Float64})*νi + (λ₆i - interval(1))*UniformScaling(interval(1))))
L₁₂ = interval.(ones(N+1))*λ₇i
L₂₁ = interval.(ones(N+1))*λ₅i
L₂₂ = diag(coefficients(project(Derivative(2), fourier, fourier,Interval{Float64}) - ν₃i*UniformScaling(interval(1))))
L_den = L₁₁.*L₂₂ - L₂₁.*L₁₂

################## Computing the nonlinear terms ######################################################
@info "Computing the nonlinear terms"
Ψ̄_big = λ₃i*Ū₁_big^2 + λ₄i*Ū₁_big - ν₁i*Ū₁_big*Ū₂_big - ν₁i*λ₁i*Ū₂_big 
Φ̄_big = exact(1) + Ū₁_big + λ₁i + ν₂i*(Ū₁_big + λ₁i)^2
Φ̄_big_inv = inv(Φ̄_big)
Ψ̄ = interval.(Float64.(inf.(Ψ̄_big),RoundDown),Float64.(sup.(Ψ̄_big),RoundUp) )
Φ̄ = interval.(Float64.(inf.(Φ̄_big),RoundDown),Float64.(sup.(Φ̄_big),RoundUp) )
Φ̄_inv= interval.(Float64.(inf.(Φ̄_big_inv),RoundDown),Float64.(sup.(Φ̄_big_inv),RoundUp) )
Ψ̄₁ = exact(1) + exact(2)*ν₂i*(Ū₁_interval + λ₁i)
Ψ̄₃ = exact(2)*λ₃i*Ū₁_interval + λ₄i - ν₁i*λ₁i*Ū₂_interval 
Ψ̄₂ = -ν₁i*Ū₁_interval - ν₁i*λ₁i


# Building g, V₁, and V₂. Note that these are not exact as we cannot represent the full inverse. We name them for demonstrational purposes.
g_big = -Ψ̄_big*Φ̄_big_inv - λ₆i*Ū₁_big - λ₇i*Ū₂_big 
g = interval.(Float64.(inf.(g_big),RoundDown),Float64.(sup.(g_big),RoundUp) )
V₁_interval = -(Φ̄*Ψ̄₃ - Ψ̄*Ψ̄₁)*Φ̄_inv^2 - λ₆i
V₂_interval = -Ψ̄₂*Φ̄_inv - λ₇i

# Computation of B.
@info "Building the B operator"
P = interval.(sqrt(2)*(vec(ones(N+1, 1))))
P[1,1] = interval(1)
P⁻¹ = (interval.(ones(N+1))./P)
DG₁₁ = project(Multiplication(V₁_interval) ,fourier, fourier, Interval{Float64})
DG₁₂ = project(Multiplication(V₂_interval) ,fourier, fourier, Interval{Float64})
M = LinearOperator(fourier^2, fourier^2, interval.(zeros(2*(N+1),2*(N+1))))
project!(component(M, 1, 1), UniformScaling(interval(1)) + DG₁₁.*(L₂₂./L_den)' - DG₁₂.*(L₂₁./L_den)')
project!(component(M, 1, 2), -DG₁₁.*(L₁₂./L_den)' + DG₁₂.*(L₁₁./L_den)')
project!(component(M, 2, 2), UniformScaling(interval(1)))

B = interval.(inv(mid.(M)))
B₁₁ = component(B,1,1)
B₁₂ = component(B,1,2)
B₁₁_adjoint = LinearOperator(fourier,fourier, coefficients(B₁₁)')
print("Computing norm of B₁₁")
norm_B₁₁ = (opnorm(LinearOperator(coefficients(P.*((B₁₁_adjoint*B₁₁)^2).*P⁻¹')),2))^(interval(1/4))
@show norm_B₁₁


################## 𝒴₀ BOUND ######################################################
@info "Computing 𝒴₀"
# Computation of the 𝒴₀ bound.
# These are the components of 𝒴₀ expanded. 
# That is, the components of ||Bᴺ(LŪ + G(Ū))||₂².
𝒴₀¹ = B₁₁*project(Derivative(2)*Ū₁_interval*νi + (λ₆i - interval(1))*Ū₁_interval + λ₇i*Ū₂_interval + g,fourier) + B₁₂*project(λ₅i*Ū₁_interval + Derivative(2)*Ū₂_interval - ν₃i*Ū₂_interval,fourier)
𝒴₀² = project(λ₅i*Ū₁_interval + Derivative(2)*Ū₂_interval - ν₃i*Ū₂_interval,fourier)

# These are the tail components of 𝒴₀ as a result of choosing N ≠ N₀ and N ≠ N₁ and having a nonlinear term
# That is the components of ||(πᴺ⁰ - πᴺ)LŪ + (πᴺ¹ - πᴺ)G(Ū)||₂²
𝒴₀¹∞ = Derivative(2)*(Ū₁_interval - project(Ū₁_interval,fourier))*νi + (λ₆i - interval(1))*(Ū₁_interval - project(Ū₁_interval,fourier)) + λ₇i*(Ū₂_interval - project(Ū₂_interval,fourier)) + g - project(g,fourier) + λ₅i*(Ū₁_interval - project(Ū₁_interval,fourier)) + Derivative(2)*(Ū₂_interval - project(Ū₂_interval,fourier)) - ν₃i*(Ū₂_interval - project(Ū₂_interval,fourier))
𝒴₀²∞ = λ₅i*(Ū₁_interval - project(Ū₁_interval,fourier)) + Derivative(2)*(Ū₂_interval - project(Ū₂_interval,fourier)) - ν₃i*(Ū₂_interval - project(Ū₂_interval,fourier))
Ω₀ = (interval(2)*di)^2
𝒴₀₁ = sqrt(norm(𝒴₀¹,2)^2 + norm(𝒴₀²,2)^2 + norm(𝒴₀¹∞,2)^2 + norm(𝒴₀²∞,2)^2)
𝒴₀₂ = norm_B₁₁*norm(Ψ̄,2)*norm(Φ̄_big_inv*(exact(1)-Φ̄_big*Φ̄_big_inv),1)/(exact(1) - norm(exact(1)-Φ̄*Φ̄_inv,1))
𝒴₀₂ = interval.(Float64.(inf.(𝒴₀₂),RoundDown),Float64.(sup.(𝒴₀₂),RoundUp) )
𝒴₀ = sqrt(Ω₀)*(𝒴₀₁ + 𝒴₀₂)
@show 𝒴₀

################## 𝒵₂ BOUND ######################################################
@info "Computing 𝒵₂"    
# Computing κ
l_den2(ξ) = νi*abs(interval(2π)*ξ)^4 + (νi*ν₃i - λ₆i+interval(1))*abs(interval(2π)*ξ)^2 - ν₃i*(λ₆i - interval(1)) - λ₅i*λ₇i
l₁₁2(ξ) = -abs(interval(2π)*ξ)^2*νi + λ₆i - interval(1)
l₁₂2(ξ) = λ₇i
l₂₁2(ξ) = λ₅i
l₂₂2(ξ) = -abs(interval(2π)*ξ)^2 - ν₃i

# Computing the bound on ||l⁻¹||_ℳ₁
norm_ℳ₁_l⁻¹_component_1_1 = _compute_ℳ₁_component(interval(-1),-ν₃i,νi,ν₃i,λ₅i,λ₆i,λ₇i,l₂₂2,l_den2)
norm_ℳ₁_l⁻¹_component_1_2 = max(abs(λ₇i/l_den2(interval(0))),abs(λ₇i/l_den2(cbrt(interval(8)*interval(π^2)*(νi*ν₃i - λ₆i + interval(1))/(interval(64)*interval(π^4))))))
norm_ℳ₁_l⁻¹_component_2_1 = max(abs(λ₅i/l_den2(interval(0))),abs(λ₅i/l_den2(cbrt(interval(8)*interval(π^2)*(νi*ν₃i - λ₆i + interval(1))/(interval(64)*interval(π^4))))))
norm_ℳ₁_l⁻¹_component_2_2 = _compute_ℳ₁_component(-νi,λ₆i - interval(1),νi,ν₃i,λ₅i,λ₆i,λ₇i,l₁₁2,l_den2)
norm_ℳ₁_l⁻¹ = sqrt(norm_ℳ₁_l⁻¹_component_1_1^2 + norm_ℳ₁_l⁻¹_component_1_2^2 + norm_ℳ₁_l⁻¹_component_2_1^2 + norm_ℳ₁_l⁻¹_component_2_2^2)
# Computing the bound on |l⁻¹|ₘ₂
_val_1_squared,_val_2_squared,_val_3_squared,_val_4_squared = integral_computation(νi,ν₃i,λ₅i,λ₆i,λ₇i)
norm_ℳ₂_l⁻¹ = maximum([sqrt(_val_1_squared + _val_2_squared) sqrt(_val_3_squared + _val_4_squared)])

κ = norm_ℳ₁_l⁻¹ * norm_ℳ₂_l⁻¹
κ₀ = interval(4)*ν₂i/(interval(4)*ν₂i - interval(1))
# Computation of the 𝒵₂ bound.
Q₁ = λ₁i*ν₁i*ν₂i + ν₁i*ν₂i*Ū₁_interval 
Q₂ = -ν₁i + λ₁i^2*ν₁i*ν₂i + interval(2)*λ₁i*ν₁i*ν₂i*Ū₁_interval + ν₁i*ν₂i*Ū₁_interval^2 
ℚ₁² = project(Multiplication(Q₁^2),fourier,fourier,Interval{Float64})
ℚ₂² = project(Multiplication(Q₂^2),fourier,fourier,Interval{Float64})

𝒵₂¹ = max(λ₁i*ν₁i*ν₂i,sqrt(opnorm(LinearOperator(coefficients(P.*(B₁₁*ℚ₁²*B₁₁_adjoint).*P⁻¹')),2) + norm(Q₁,1)^2))
𝒵₂² = max(abs(-ν₁i+λ₁i*ν₁i*ν₂i),sqrt(opnorm(LinearOperator(coefficients(P.*(B₁₁*ℚ₂²*B₁₁_adjoint).*P⁻¹')),2) + norm(Q₂,1)^2))

υ⁰¹₀₀,υ¹⁰₀₀,υ¹¹₀₀,υ²⁰₀₀,υ²¹₀₀,υ³⁰₀₀,υ⁴⁰₀₀,Q₀₁,Q₁₁,Q₁₀,Q₂₁,Q₂₀,Q₃₀,Q₄₀ = _compute_qijs(Ū₁_interval,Ū₂_interval,ν₁i,ν₂i,λ₁i,λ₃i,λ₄i)

ℚ₀₁² = project(Multiplication(Q₀₁^2),fourier,fourier,Interval{Float64})
ℚ₁₀² = project(Multiplication(Q₁₀^2),fourier,fourier,Interval{Float64})
ℚ₁₁² = project(Multiplication(Q₁₁^2),fourier,fourier,Interval{Float64})
ℚ₂₀² = project(Multiplication(Q₂₀^2),fourier,fourier,Interval{Float64})
ℚ₂₁² = project(Multiplication(Q₂₁^2),fourier,fourier,Interval{Float64})
ℚ₃₀² = project(Multiplication(Q₃₀^2),fourier,fourier,Interval{Float64})
ℚ₄₀² = project(Multiplication(Q₄₀^2),fourier,fourier,Interval{Float64})

𝒵₂⁰¹ = max(abs(υ⁰¹₀₀),sqrt(opnorm(LinearOperator(coefficients(P.*(B₁₁*ℚ₀₁²*B₁₁_adjoint).*P⁻¹')),2) + norm(Q₀₁,1)^2))
𝒵₂¹⁰ = max(abs(υ¹⁰₀₀),sqrt(opnorm(LinearOperator(coefficients(P.*(B₁₁*ℚ₁₀²*B₁₁_adjoint).*P⁻¹')),2) + norm(Q₁₀,1)^2))
𝒵₂¹¹ = max(abs(υ¹¹₀₀),sqrt(opnorm(LinearOperator(coefficients(P.*(B₁₁*ℚ₁₁²*B₁₁_adjoint).*P⁻¹')),2) + norm(Q₁₁,1)^2))
𝒵₂²⁰ = max(abs(υ²⁰₀₀),sqrt(opnorm(LinearOperator(coefficients(P.*(B₁₁*ℚ₂₀²*B₁₁_adjoint).*P⁻¹')),2) + norm(Q₂₀,1)^2))
𝒵₂²¹ = max(abs(υ²¹₀₀),sqrt(opnorm(LinearOperator(coefficients(P.*(B₁₁*ℚ₂₁²*B₁₁_adjoint).*P⁻¹')),2) + norm(Q₂₁,1)^2))
𝒵₂³⁰ = max(abs(υ³⁰₀₀),sqrt(opnorm(LinearOperator(coefficients(P.*(B₁₁*ℚ₃₀²*B₁₁_adjoint).*P⁻¹')),2) + norm(Q₃₀,1)^2))
𝒵₂⁴⁰ = max(abs(υ⁴⁰₀₀),sqrt(opnorm(LinearOperator(coefficients(P.*(B₁₁*ℚ₄₀²*B₁₁_adjoint).*P⁻¹')),2) + norm(Q₄₀,1)^2))

𝒵₂ = κ₀^2*κ*(𝒵₂¹ + 𝒵₂²*norm_ℳ₂_l⁻¹*r₀ + κ₀^2*(𝒵₂⁰¹ + 𝒵₂¹⁰ + 𝒵₂¹¹*norm_ℳ₂_l⁻¹*r₀ + 𝒵₂²⁰*norm_ℳ₂_l⁻¹*r₀ + 𝒵₂²¹*norm_ℳ₂_l⁻¹^2*r₀^2 + 𝒵₂³⁰*norm_ℳ₂_l⁻¹^2*r₀^2 + 𝒵₂⁴⁰*norm_ℳ₂_l⁻¹^3*r₀^3))
@show 𝒵₂
################## 𝒵₁ BOUND ###################################################### 
#These are the true Vⱼᴺ terms.
V₁ᴺ_interval = project(V₁_interval,CosFourier(N,π/di))
V₂ᴺ_interval = project(V₂_interval,CosFourier(N,π/di))
L⁻¹_norm = φ(norm_ℳ₁_l⁻¹_component_1_1, norm_ℳ₁_l⁻¹_component_1_2, norm_ℳ₁_l⁻¹_component_2_1, norm_ℳ₁_l⁻¹_component_2_2)

𝒵_∞₁ = norm(V₁_interval - V₁ᴺ_interval,1) + norm(Φ̄*Ψ̄₃-Ψ̄*Ψ̄₁,1)*norm(Φ̄_inv^2*(exact(1)-Φ̄^2*Φ̄_inv^2),1)/(exact(1) - norm(exact(1)-Φ̄^2*Φ̄_inv^2,1))
𝒵_∞₂ = norm(V₂_interval - V₂ᴺ_interval,1) + norm(Ψ̄₂,1)*norm(Φ̄_inv*(exact(1)-Φ̄*Φ̄_inv),1)/(exact(1) - norm(exact(1)-Φ̄*Φ̄_inv,1))
𝒵_∞ = sqrt(𝒵_∞₁^2 + 𝒵_∞₂^2)

# Computation of Z₁.
@info "Computing Z₁"  
l₁₁2N = -(interval(2N+1)*π/di)^2*νi + λ₆i - interval(1)
l₁₂2N = λ₇i
l₂₁2N = λ₅i
l₂₂2N = -(interval(2N+1)*π/di)^2 - ν₃i
l_den2N = l₁₁2N*l₂₂2N - l₁₂2N*l₂₁2N
Z₁₁ = abs(l₂₂2N/l_den2N) * norm(V₁ᴺ_interval,1) + abs(l₂₁2N/l_den2N) * norm(V₂ᴺ_interval,1)
Z₁₂ = abs(l₁₂2N/l_den2N) * norm(V₁ᴺ_interval,1) + abs(l₁₁2N/l_den2N) * norm(V₂ᴺ_interval,1)

fourier2 = CosFourier(2N,π/di)
fourier3 = CosFourier(3N,π/di)

M_2N_3N = LinearOperator(fourier2^2, fourier3^2, interval.(zeros(2*(3N+1),2*(2N+1))))
L₁₁_2N = diag(coefficients(project(Derivative(2), fourier2, fourier2,Interval{Float64})*νi + (λ₆i - interval(1))*UniformScaling(interval(1))))
L₁₂_2N = interval.(ones(2N+1))*λ₇i
L₂₁_2N = interval.(ones(2N+1))*λ₅i
L₂₂_2N = diag(coefficients(project(Derivative(2), fourier2, fourier2,Interval{Float64}) - ν₃i*UniformScaling(interval(1)))) 
L_den_2N = L₁₁_2N.*L₂₂_2N - L₂₁_2N.*L₁₂_2N

𝕍₁ᴺ_2N_3N = project(Multiplication(V₁ᴺ_interval),fourier2,fourier3,Interval{Float64})
𝕍₂ᴺ_2N_3N = project(Multiplication(V₂ᴺ_interval),fourier2,fourier3,Interval{Float64})
project!(component(M_2N_3N, 1, 1), UniformScaling(interval(1)) + 𝕍₁ᴺ_2N_3N.*(L₂₂_2N./L_den_2N)' - 𝕍₂ᴺ_2N_3N.*(L₂₁_2N./L_den_2N)')
project!(component(M_2N_3N, 1, 2), -𝕍₁ᴺ_2N_3N.*(L₁₂_2N./L_den_2N)' + 𝕍₂ᴺ_2N_3N.*(L₁₁_2N./L_den_2N)')
project!(component(M_2N_3N, 2, 2), UniformScaling(interval(1)))

B = LinearOperator(fourier^2,fourier^2, [coefficients(B₁₁) coefficients(B₁₂) ; interval.(zeros(N+1,N+1)) interval.(Diagonal(ones(N+1)))])
B_3N = project(B,fourier3^2,fourier3^2)
component(B_3N,1,1)[N+1:end,N+1:end] .= Diagonal(interval.(ones(2N)))
component(B_3N,2,2)[N+1:end,N+1:end] .= Diagonal(interval.(ones(2N)))

P_2N = interval.(sqrt(2)*(vec(ones(2N+1, 1))))
P_2N[1,1] = interval(1)
P_2N⁻¹ = (interval.(ones(2N+1))./P_2N)
P_3N = interval.(sqrt(2)*(vec(ones(3N+1, 1))))
P_3N[1,1] = interval(1)

Z₀ = opnorm(LinearOperator(coefficients([P_3N ; P_3N].*(UniformScaling(interval(1)) - B_3N*M_2N_3N).*[P_2N⁻¹ ; P_2N⁻¹]')),2)

Z₁ = sqrt(Z₀^2 + Z₁₁^2 + Z₁₂^2)
@show Z₁ 
# Computation of 𝒵ᵤ.
@info "Computing 𝒵ᵤ"
# Computing the constants Cⱼ and a.
a,C₁,C₂,C₃,C₄ = _compute_C_a(νi,ν₃i,λ₅i,λ₆i,λ₇i)

# Building the Fourier series of E.
E = Sequence(CosFourier(2N,π/d), interval.(zeros(2N+1)))
for n = 0:2N
    E[n] = interval(2)*a* interval(-1)^interval(n)/(interval(4)*a^2 + (interval(n)*π/di)^2)
end

Cd = interval(4)*di + interval(4)*exp(-a*di)/(a*(interval(1)-exp(-interval(3/2)*a*di))) + interval(2)/(a*(interval(1)-exp(-interval(2)*a*di)))

# Computing the inner products.
EV₁ᴺ = _conv_small(E,V₁ᴺ_interval,N)
EV₂ᴺ = _conv_small(E,V₂ᴺ_interval,N)
V₁ᴺ_inner_prodEV₁ᴺ = abs(coefficients(P.*V₁ᴺ_interval)'*coefficients(P.*EV₁ᴺ))
V₂ᴺ_inner_prodEV₂ᴺ = abs(coefficients(P.*V₂ᴺ_interval)'*coefficients(P.*EV₂ᴺ))

𝒵ᵤ₁₁ = sqrt(C₁^2*(interval(1)-exp(-interval(4)*a*di))/a * V₁ᴺ_inner_prodEV₁ᴺ)
𝒵ᵤ₁₂ = sqrt(C₂^2*(interval(1)-exp(-interval(4)*a*di))/a * V₂ᴺ_inner_prodEV₂ᴺ)
𝒵ᵤ₁₃ = sqrt(C₃^2*(interval(1)-exp(-interval(4)*a*di))/a * V₁ᴺ_inner_prodEV₁ᴺ)
𝒵ᵤ₁₄ = sqrt(C₄^2*(interval(1)-exp(-interval(4)*a*di))/a * V₂ᴺ_inner_prodEV₂ᴺ)
𝒵ᵤ₁ = sqrt((𝒵ᵤ₁₁ + 𝒵ᵤ₁₂)^2 + (𝒵ᵤ₁₃ + 𝒵ᵤ₁₄)^2)

𝒵ᵤ₂₁ = sqrt(𝒵ᵤ₁₁^2 + Cd*C₁^2*(exp(-interval(2)*a*di)-exp(-interval(6)*a*di))*V₁ᴺ_inner_prodEV₁ᴺ)
𝒵ᵤ₂₂ = sqrt(𝒵ᵤ₁₂^2 + Cd*C₂^2*(exp(-interval(2)*a*di)-exp(-interval(6)*a*di))*V₂ᴺ_inner_prodEV₂ᴺ)
𝒵ᵤ₂₃ = sqrt(𝒵ᵤ₁₃^2 + Cd*C₃^2*(exp(-interval(2)*a*di)-exp(-interval(6)*a*di))*V₁ᴺ_inner_prodEV₁ᴺ)
𝒵ᵤ₂₄ = sqrt(𝒵ᵤ₁₄^2 + Cd*C₄^2*(exp(-interval(2)*a*di)-exp(-interval(6)*a*di))*V₂ᴺ_inner_prodEV₂ᴺ)
𝒵ᵤ₂ = sqrt((𝒵ᵤ₂₁ + 𝒵ᵤ₂₂)^2 + (𝒵ᵤ₂₃ + 𝒵ᵤ₂₄)^2)
𝒵ᵤ = sqrt(𝒵ᵤ₁^2 + 𝒵ᵤ₂^2)
@show 𝒵ᵤ

𝒵₁ = Z₁ + norm_B₁₁*(𝒵ᵤ + L⁻¹_norm*𝒵_∞)
@show 𝒵₁

################## Computer Assisted Proof ######################################################
r_min = sup((interval(1) - 𝒵₁ - sqrt((interval(1) - 𝒵₁)^2 - interval(2)*𝒴₀*𝒵₂))/𝒵₂)
r_max = inf((interval(1) - 𝒵₁ + sqrt((interval(1) - 𝒵₁)^2 - interval(2)*𝒴₀*𝒵₂))/𝒵₂)
CAP(sup(𝒴₀),sup(𝒵₁),sup(𝒵₂))
