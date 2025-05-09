using DifferentialEquations
using Plots
using JSON
using Dates
using Random
using Base.Filesystem

# Define the FitzHugh-Nagumo model
function fitzhugh_nagumo!(du, u, p, t)
    v, w = u
    a, b, c, I = p
    du[1] = c * (v - (v^3)/3 - w + I)  # dv/dt
    du[2] = (v + a - b * w) / c        # dw/dt
end

# Default parameters
const DEFAULT_PARAMS = Dict(
    :a => 0.7,
    :b => 0.8,
    :c => 3.0,
    :I => 0.5,
    :tspan => (0.0, 100.0),
    :u0 => [0.1, 0.0],
    :dt => 0.01
)

# Default configuration
const DEFAULT_CONFIG = Dict(
    :base_dir => "simulations",
    :data_file => "solution.json",
    :plot_file => "plot.png",
    :parallel => false  # Toggle parallel processing
)

struct SimulationManager
    params::Dict
    config::Dict
end

function SimulationManager(params; config=Dict())
    cfg = merge(DEFAULT_CONFIG, config)
    SimulationManager(params, cfg)
end

function run_simulations(manager::SimulationManager)
    cfg = manager.config

    # Create base directory
    mkpath(cfg[:base_dir])

    # Set up base problem
    params = merge(DEFAULT_PARAMS, manager.params)
    p = [params[:a], params[:b], params[:c], params[:I]]
    prob = ODEProblem(fitzhugh_nagumo!, params[:u0], params[:tspan], p)

    # Solve the problem
    sol = solve(prob, Tsit5(), saveat=params[:dt])

    # Create unique run directory
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    run_id = string(randstring(8))
    run_dir = joinpath(cfg[:base_dir], "run_$(timestamp)_$(run_id)")
    mkpath(run_dir)

    # Save solution data
    sol_data = Dict(
        :t => sol.t,
        :v => [u[1] for u in sol.u],
        :w => [u[2] for u in sol.u],
        :params => params,
        :config => cfg
    )
    open(joinpath(run_dir, cfg[:data_file]), "w") do f
        JSON.print(f, sol_data, 4)
    end

    # Generate and save plot
    plt = plot(sol, idxs=(1,2), xlabel="v", ylabel="w", title="FitzHugh-Nagumo Phase Plane")
    savefig(plt, joinpath(run_dir, cfg[:plot_file]))
end

# Example usage
function main()
    # Define multiple parameter sets for different runs
    params = Dict(
        :a => 0.75, 
        :I => 0.6, 
        :tspan => (0.0, 200.0)
    )

    # Custom configuration
    config = Dict(
        :base_dir => "fhn_simulations",
        :data_file => "fhn_data.json",
        :plot_file => "fhn_plot.png",
        :parallel => false
    )

    # Create and run simulation manager
    manager = SimulationManager(params, config=config)
    run_simulations(manager)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end