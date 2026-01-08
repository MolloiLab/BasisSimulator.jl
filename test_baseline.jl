#!/usr/bin/env julia
# Quick test to verify current pipeline produces valid reconstruction

using Pkg
Pkg.activate(@__DIR__)

println("Loading packages...")
using Statistics

println("Including notebook code...")
# Load the notebook by executing all code cells (skipping markdown)
include("mwe_imaging_chain.jl")

println("\n=== BASELINE TEST ===")
println("Phantom size: $(size(PHANTOM.material_ids))")
println("Number of materials: $(length(unique(PHANTOM.material_ids)))")
println("Grid: $(PHANTOM.grid.nx) × $(PHANTOM.grid.ny) × $(PHANTOM.grid.nz)")
println("\nSource energy range: $(minimum(SOURCE_120.energies)) - $(maximum(SOURCE_120.energies)) keV")
println("Total photons: $(sum(SOURCE_120.photons))")
println("\nGeometry: $(length(GEOMETRY.angles)) projections")
println("Detector: $(GEOMETRY.n_rows) × $(GEOMETRY.n_cols)")

println("\n=== CHECKING RECONSTRUCTION ===")
if @isdefined vol_hu
    println("Reconstruction exists!")
    println("Volume size: $(size(vol_hu))")
    println("HU range: $(minimum(vol_hu)) to $(maximum(vol_hu))")
    println("HU mean: $(mean(vol_hu))")
    println("HU std: $(std(vol_hu))")

    # Check for NaN/Inf
    if any(isnan, vol_hu) || any(isinf, vol_hu)
        println("⚠️  WARNING: Found NaN or Inf values!")
    else
        println("✅ Reconstruction valid (no NaN/Inf)")
    end
else
    println("❌ Reconstruction not found - need to run simulation")
end

println("\n=== TEST COMPLETE ===")
