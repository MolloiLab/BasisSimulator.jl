"""
# Noise Physics Module

Implements realistic noise models for CT:
- Poisson (quantum) noise
- Electronic noise (Gaussian)
- Combined noise models

## References
- Barrett & Myers (2004) Foundations of Image Science
- Wagner et al. (1999) Med Phys 26(11):2392-2402
"""

using Random
using Distributions

function add_poisson_noise(
        signal::Array{Float64},
        dose_scale::Float64 = 1.0;
        rng::AbstractRNG = Random.default_rng()
    )::Array{Float64}
    
    scaled_signal = signal .* dose_scale
    scaled_signal = max.(scaled_signal, 1e-10)
    
    noisy = similar(signal)
    for i in eachindex(signal)
        if scaled_signal[i] > 0
            lambda = scaled_signal[i]
            noisy[i] = rand(rng, Poisson(lambda)) / dose_scale
        else
            noisy[i] = 0.0
        end
    end
    
    return noisy
end

function add_electronic_noise(
        signal::Array{Float64},
        sigma::Float64;
        rng::AbstractRNG = Random.default_rng()
    )::Array{Float64}
    
    noise = randn(rng, size(signal)...) .* sigma
    return signal .+ noise
end

function add_ct_noise(
        signal::Array{Float64},
        dose_scale::Float64 = 1.0,
        electronic_sigma::Float64 = 10.0;
        rng::AbstractRNG = Random.default_rng()
    )::Array{Float64}
    
    noisy = add_poisson_noise(signal, dose_scale, rng=rng)
    noisy = add_electronic_noise(noisy, electronic_sigma, rng=rng)
    return noisy
end

export add_poisson_noise, add_electronic_noise, add_ct_noise
