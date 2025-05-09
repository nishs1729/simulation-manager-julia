using Dates
using HDF5
using JSON
using Logging
using Random
using DifferentialEquations
using Plots

# Abstract type for simulation managers
abstract type AbstractSimulationManager end

# Concrete simulation manager struct
mutable struct SimulationManager
    config::Dict
    params::Dict
    trial::Int
    data_loc::String
    sim_dir::String
    sim_path::String
    hdf5_path::String
    default_params::Dict

    function SimulationManager(config, seed, log_info; default_params=Dict())
        # Validate default_params
        if !isa(default_params, Dict)
            error("default_params must be a dictionary")
        end

        # Set seed
        trial = isnothing(seed) ? 1 : seed
        Random.seed!(trial)

        # Initialize struct
        sim = new()

        # Set config and params
        sim.default_params = default_params
        sim = set_config!(sim, config)
        sim.trial = trial

        # Set attributes from params
        for (key, value) in sim.params
            println(key, value)
            setfield!(sim, Symbol(key), value)
        end

        # Setup directories and logging
        setup_directories!(sim, log_info)
        save_config!(sim)

        return sim
    end
end

function set_config!(sim::SimulationManager, config)
    # Set configuration from dict or JSON file
    if isa(config, Dict)
        sim.config = config
    elseif isa(config, String) && isfile(config)
        sim.config = JSON.parsefile(config)
    else
        error("Config must be a dictionary or a valid path to a JSON file")
    end

    # Merge default and provided params
    sim.params = copy(sim.default_params)
    merge!(sim.params, get(sim.config, "params", Dict()))
    return sim
end

function setup_directories!(sim::SimulationManager, log_info)
    # Set data location and simulation directory
    sim.data_loc = sim.config["data_loc"]
    if contains(lowercase(log_info), "test")
        sim.sim_dir = "test"
    elseif haskey(sim.config, "sim_dir") && !isempty(sim.config["sim_dir"])
        sim.sim_dir = sim.config["sim_dir"]
    else
        sim.sim_dir = "sim_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
    end

    sim.sim_path = joinpath(sim.data_loc, sim.sim_dir)
    mkpath(sim.sim_path)

    # Setup logging
    log_level = contains(lowercase(log_info), "debug") ? Logging.Debug : Logging.Info
    log_file = joinpath(sim.sim_path, "sim.log")
    logger = SimpleLogger(open(log_file, "w"), log_level)
    global_logger(logger)

    # Create Readme.md
    readme_path = joinpath(sim.sim_path, "Readme.md")
    open(readme_path, "w") do f
        write(f, sim.config["description"])
    end

    # Create HDF5 file
    sim.hdf5_path = joinpath(sim.sim_path, "data_$(sim.trial).hdf5")
    h5open(sim.hdf5_path, "w") do f end

    @info "Simulation directory: '$(sim.sim_path)'"
end

function save_config!(sim::SimulationManager)
    # Save config to JSON
    sim.config["params"] = sim.params
    config_path = joinpath(sim.sim_path, "config.json")
    open(config_path, "w") do f
        JSON.print(f, sim.config, 4)
    end
    @info "Saved config to '$config_path'"
end

# Define the run method as an interface
function run!(sim::AbstractSimulationManager)
    error("Method run! must be implemented for $(typeof(sim))")
end

#########################################################
# FHN simulation struct
mutable struct FHN <: AbstractSimulationManager
    sim_mgr::SimulationManager
    t::Vector{Float64}
    v::Union{Vector{Float64}, Nothing}
    w::Union{Vector{Float64}, Nothing}
    method::String

    function FHN(config; seed=nothing, log_info="")
        sim_mgr = SimulationManager(config, seed, log_info; default_params=Dict(
            "a" => 0.7,
            "b" => 0.8,
            "tau" => 12.5,
            "I" => 0.5,
            "dt" => 0.01,
            "y0" => [0.1, 0.0],
            "tend" => 100.0
        ))

        t = collect(0:sim_mgr.params["dt"]:sim_mgr.params["tend"])
        method = get(config, "method", "RK45")
        new(sim_mgr, t, nothing, nothing, method)
    end
end

function fhn_system(t, y, p)
    v, w = y
    a, b, tau, I = p
    dv = v - (v^3) / 3 - w + I
    dw = (v + a - b * w) / tau
    return [dv, dw]
end

function run!(fhn::FHN)
    params = (fhn.sim_mgr.params["a"], fhn.sim_mgr.params["b"], 
              fhn.sim_mgr.params["tau"], fhn.sim_mgr.params["I"])
    prob = ODEProblem(fhn_system, fhn.sim_mgr.params["y0"], 
                      (0.0, fhn.sim_mgr.params["tend"]), params)
    sol = solve(prob, getfield(DifferentialEquations, Symbol(fhn.method))(), 
                t=fhn.t, saveat=fhn.t)
    fhn.v = sol[1, :]
    fhn.w = sol[2, :]
end

function save_data!(fhn::FHN)
    h5open(fhn.sim_mgr.hdf5_path, "w") do f
        f["time"] = fhn.t
        f["v"] = fhn.v
        f["w"] = fhn.w
    end
    @info "Saved simulation data to $(fhn.sim_mgr.hdf5_path)"
end

function plot_results(fhn::FHN)
    # Time series plot
    p1 = plot(fhn.t, fhn.v, label="v (membrane potential)", xlabel="Time", ylabel="Variables")
    plot!(p1, fhn.t, fhn.w, label="w (recovery variable)")

    # Phase portrait
    v_nullcline = range(-2, 2, length=500)
    w_v_nullcline = v_nullcline .- (v_nullcline .^ 3) ./ 3 .+ fhn.sim_mgr.params["I"]
    w_w_nullcline = (v_nullcline .+ fhn.sim_mgr.params["a"]) ./ fhn.sim_mgr.params["b"]

    p2 = plot(fhn.v, fhn.w, label="Trajectory", xlabel="v (membrane potential)", 
              ylabel="w (recovery variable)")
    plot!(p2, v_nullcline, w_v_nullcline, label="dv/dt = 0", linestyle=:dash, color=:red)
    plot!(p2, v_nullcline, w_w_nullcline, label="dw/dt = 0", linestyle=:dash, color=:blue)

    # Combine plots
    plt = plot(p1, p2, layout=(2,1), size=(600, 800))
    display(plt)
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    config = Dict(
        "description" => "Testing",
        "sim_dir" => "dfgdfg",
        "data_loc" => "data/",
        "params" => Dict(
            "tend" => 200.0,
            "b" => 1.0,
            "y0" => [0.1, 0.5]
        )
    )

    model = FHN(config, seed=42, log_info="")
    # run!(model)
    # plot_results(model)
    # save_data!(model)
end