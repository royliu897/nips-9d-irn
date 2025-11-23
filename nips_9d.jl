using DrWatson
@quickactivate

using Distributed, Sunny, WGLMakie, LatinHypercubeSampling

addprocs(16) # Use 16 worker processes

# Using these packages across all 16 workers
@everywhere using DrWatson
@everywhere @quickactivate
@everywhere using Sunny, LinearAlgebra, HDF5, LatinHypercubeSampling

# Define the material's unit cell
@everywhere function nips_crystal() 
    latvecs = lattice_vectors(5.8196, 10.084, 6.8959, 90, 106.221, 90)
    positions = [[0, 1/3, 0]]
    Crystal(latvecs, positions, 12)
end

# Generate Latin Hypercube sampling plan
function generate_lhs_plan(n_samples)
    
    # Define ranges for the 9 parameters
    # Format: (Min, Max)
    ranges = [
        (-0.02, 0.0),   # Ax   (Base -0.01)
        (0.0, 0.42),    # Az   (Base 0.21)
        (-5.4, 0.0),    # J1a  (Base -2.7)
        (-4.0, 0.0),    # J1b  (Base -2.0)
        (0.0, 0.4),     # J2a  (Base 0.2)
        (0.0, 0.4),     # J2b  (Base 0.2)
        (0.0, 27.8),    # J3a  (Base 13.9)
        (0.0, 27.8),    # J3b  (Base 13.9)
        (-0.76, 0.0)    # J4   (Base -0.38)
    ]
    
    println("Generating Latin Hypercube Plan for $n_samples samples...")
    
    # Generate (n_samples x 9) matrix of numbers from evenly spaced bins, each column is a Hamiltonian parameter 
    plan = LatinHypercubeSampling.randomLHC(n_samples, 9)
    
    # Scale plan to actual physical ranges for each Hamiltonian parameter
    scaled_plan = LatinHypercubeSampling.scaleLHC(plan, ranges)
    
    return scaled_plan
end

# Physics, defines Hamiltonian
@everywhere function nips_system(; Ax, Az, J1a, J1b, J2a, J2b, J3a, J3b, J4)
    crystal = nips_crystal()
    sys = System(crystal, [1 => Moment(; s=1, g=2)], :dipole_uncorrected)
    S = spin_matrices(Inf)
    
    set_onsite_coupling!(sys, S -> Ax*S[1]^2 + Az*S[3]^2, 1)
    set_exchange!(sys, J1a, Bond(2, 1, [0, 0, 0]))
    set_exchange!(sys, J1b, Bond(2, 3, [0, 0, 0]))
    set_exchange!(sys, J2a,  Bond(2, 2, [1, 0, 0]))
    set_exchange!(sys, J2b,  Bond(1, 4, [0, 0, 0]))
    set_exchange!(sys, J3a,  Bond(1, 3, [0, 0, 0]))
    set_exchange!(sys, J3b,  Bond(2, 3, [1, 0, 0]))
    set_exchange!(sys, J4,   Bond(1, 1, [0, 0, 1]))

    randomize_spins!(sys)
    minimize_energy!(sys; maxiters=10_000)

    return sys
end

# Simulation loop for one sample
@everywhere function compute_one_sample(row_data)
    # Unpack row_data (index, params_vector)
    idx, p_vec = row_data
    
    # Map vector to named parameters
    params = (
        Ax=p_vec[1], Az=p_vec[2], 
        J1a=p_vec[3], J1b=p_vec[4], 
        J2a=p_vec[5], J2b=p_vec[6], 
        J3a=p_vec[7], J3b=p_vec[8], 
        J4=p_vec[9]
    )

    # Fixed configuration
    energies = range(0, 150, 150)
    kernel = gaussian(; fwhm=4)
    rotations = [([0,0,1], π/3), ([0,0,1], 2π/3), ([0,0,1], 0.0)]
    weights = [1,1,1]
    n_q_points = 2500 

    # Build System
    sys = nips_system(; params...)
    
    measure = ssf_perp(sys; formfactors=[1=>FormFactor("Ni2")])
    swt = SpinWaveTheory(sys; measure=measure, regularization=1e-6)

    qs_rlu = [rand(3) .- 0.5 for _ in 1:n_q_points] #Pick random points in Brillouin Zone
    
    try
        res = domain_average(sys.crystal, qs_rlu; 
                            rotations=rotations, weights=weights) do path_rot
            intensities(swt, path_rot; energies, kernel)
        end

        data = res.data
        return (params=params, qs_rlu=Float32.(reduce(hcat, qs_rlu)'), energies=Float32.(collect(energies)), data=Float32.(data))

    catch e
        @warn "Sample $idx failed." exception=(e, catch_backtrace())
        return nothing # Handle failure gracefully in main loop
    end
end

# ------------------------------------------------------
# EXECUTION
# ------------------------------------------------------

nsamples = 20000
lhs_matrix = generate_lhs_plan(nsamples)

# Create input pairs (index, parameter_row)
inputs = [(i, lhs_matrix[i, :]) for i in 1:nsamples]

println("Distributing jobs...")
results = pmap(compute_one_sample, inputs)

# Filter failed jobs
samples = filter(!isnothing, results)
println("Success rate: $(length(samples)) / $nsamples")

# ------------------------------------------------------
# FILE SAVING
# ------------------------------------------------------

#Write HDF5 file to Scratch for faster I/O
output_dir = joinpath(ENV["SCRATCH"], "nips_prelim")
mkpath(output_dir)
output_file = joinpath(output_dir, "nips_9d_data_gen.h5")
println("Writing to $output_file...")

h5open(output_file, "w") do f
    f["energies"] = samples[1].energies

    for (i, s) in enumerate(samples)
        g = create_group(f, "sample_$i")
        g["qs_rlu"] = s.qs_rlu
        g["data"]   = s.data

        p = create_group(g, "params")
        for (k, v) in pairs(s.params)
            p[String(k)] = Float32(v)
        end
    end
end
println("Done!")

# Generates nsamples x n_q_points * energies data points:
# 20,000 * 2,500 * 150 = 7.5 billion data points, ~30gb
