"""
    Cliques

This module provides functionality for finding the largest cliques in a graph using various
algorithms, including Bron-Kerbosch pivoting, subgraph removal, and greedy methods. It also
includes utilities for validating cliques, generating random graphs containing cliques, and
scheduling the execution of these algorithms in a distributed manner.

# Examples
```julia-repl
julia> include("./cliques.jl")
main (generic function with 1 method)

julia> using .Cliques, .Cliques.CliquesIO, .Cliques.Schedule

julia> write(
           "./test.clq",
           \"""
           c Mycielsky-3 graph;
           c |V|=5, omega(M3)=3, Chi(M3)=3
           c
           p edge 5 5
           e 1 2
           e 1 3
           e 1 4
           e 2 3
           e 3 5
           e 4 5
           \"""
       )
102

julia> A = readclq("./test.clq") # read the CLQ file
5×5 BitMatrix:
 0  1  1  1  0
 1  0  1  0  0
 1  1  0  0  1
 1  0  0  0  1
 0  0  1  1  0

julia> lclq = largestcliques(A) # find the largest cliques
Main.Cliques.MaximalCliques(:BKPivoting, false, 3, [[1, 2, 3]])

julia> report_largest(lclq) # print readable largest cliques information
Largest cliques found in the CLQ file:
  By: Bron-Kerbosch pivoting (OPTIMAL SOLUTION)
  Size: ω = 3
  Number of cliques found: 1
  Vertices:
    [1, 2, 3]

julia> lclqs = scheduler(A) # intelligently choose algorithms and exploit parallel computing
Searching for the largest cliques in the CLQ file...
Time remaining: 57 sec
Bron-Kerbosch pivoting:
  ✓ Solution found on worker 1.
    ω = 3; found: 1
Subgraph removal:
  No tasks scheduled.
Greedy:
  ✓ Solution found on worker 1.
    ω = 3; found: 1

Main.Cliques.MaximalCliques(:BKPivoting, false, 3, [[1, 2, 3]])
```

Waylon Wu
2nd June 2025
"""
module Cliques

export largestcliques, complexity, validate, generate

import Random

# The main function enter point that can be called from the command line. It needs the path
# to the CLQ file as the argument. It reads the graph from the file, runs the scheduler, and
# reports the largest cliques found.
function (@main)(args)
    A = Cliques.CliquesIO.readclq(args[1]) # read the graph from the file
    lclq = Cliques.Schedule.scheduler(A, 55.0, 0.9, args[1], stderr) # run the scheduler
    Cliques.CliquesIO.report_largest(lclq, args[1]) # report the largest cliques found
end

"""
    MaximalCliques

A structure to hold the results of maximal clique finding algorithms.

# Fields
- `alg::Symbol`: The algorithm used to find the cliques. The symbols consistent with the
    name of the modules that those algorithms are implemented in. Possible values are:
    `:BKPivoting`, `:SubgraphRemoval`, and `:Greedy`.
- `timeout::Bool`: A flag indicating whether the algorithm was terminated due to timeout.
- `omega::Int`: The size of the largest cliques found (ω).
- `cliques::Vector{Vector{Int}}`: A vector of vectors, where each inner vector contains the
    indices of the vertices in a maximal clique.
"""
struct MaximalCliques
    alg::Symbol
    timeout::Bool
    omega::Int
    cliques::Vector{Vector{Int}}
end

MaximalCliques(
    soltuple::Tuple{Int,Vector{Vector{Int}}}, alg::Symbol, timeout::Bool=false
) = MaximalCliques(alg, timeout, soltuple[1], soltuple[2])

"""
    largestcliques(A::BitMatrix, alg::Symbol=:BKPivoting)::MaximalCliques

Finds the largest cliques in a graph represented by a bit matrix `A` using the specified
algorithm `alg`. `alg` is a symbol that consistent with the name of the modules that those
algorithms are implemented in. Possible values are: `:BKPivoting`, `:SubgraphRemoval`, and
`:Greedy`. The default is `:BKPivoting`.

# Examples
```julia-repl
julia> A = BitMatrix([
           0  1  1  1  0
           1  0  1  0  0
           1  1  0  0  1
           1  0  0  0  1
           0  0  1  1  0
       ]);


julia> largestcliques(A)
Main.Cliques.MaximalCliques(:BKPivoting, false, 3, [[1, 2, 3]])

julia> largestcliques(A, :Greedy)
Main.Cliques.MaximalCliques(:Greedy, false, 3, [[1, 2, 3]])
```
"""
largestcliques(A::BitMatrix, alg::Symbol=:BKPivoting)::MaximalCliques = eval(alg).largestcliques(A)

"""
    complexity(A::BitMatrix, alg::Symbol=:BKPivoting)::Float64

Approximate runtime time of the algorithm `alg` on the bit matrix `A`. The default algorithm
is `:BKPivoting`. For `:BKPivoting`, it only returns 3 possible values: `1.0`, `10.0`, and
`Inf`. The value `1.0` and `10.0` indicates that the algorithm will likely to time in O(1)
seconds and O(10) seconds, respectively. The value `Inf` indicates that the algorithm will
likely to take more than 100 seconds to finish.

!!! warning
    This approximation may be highly unrealiable and only serves as a reference or guess.

# Examples
```julia-repl
julia> A = BitMatrix([
           0  1  1  1  0
           1  0  1  0  0
           1  1  0  0  1
           1  0  0  0  1
           0  0  1  1  0
       ]);

julia> complexity(A)
1.0

julia> complexity(A, :SubgraphRemoval)
2.7796111265102192
```
"""
complexity(A::BitMatrix, alg::Symbol=:BKPivoting)::Float64 = eval(alg).complexity(A)

"""
    validate(A::BitMatrix, cliques::MaximalCliques)::Bool

Validate all cliques claimed by `cliques` in the bit matrix `A` are indeed cliques.

# Examples
```julia-repl
julia> A = BitMatrix([
           0  1  1  1  0
           1  0  1  0  0
           1  1  0  0  1
           1  0  0  0  1
           0  0  1  1  0
       ]);

julia> lclq = largestcliques(A)
Main.Cliques.MaximalCliques(:BKPivoting, false, 3, [[1, 2, 3]])

julia> validate(A, lclq)
true
```
"""
function validate(A::BitMatrix, cliques::MaximalCliques)::Bool
    flag = all(isequal(cliques.omega) ∘ length, cliques.cliques) # all cliques have size ω
    for clq in cliques.cliques
        @inbounds clqmat = A[clq, clq] # submatrix of the clique
        foreach(i -> @inbounds(clqmat[i, i] = true), axes(clqmat, 1)) # complete matrix
        flag &= all(clqmat)
    end
    return flag
end

"""
    generate(
        n::Int, lbound::Int, p::Float64=0.1; rng::Random.AbstractRNG=Random.default_rng()
    )::Tuple{BitMatrix,Vector{Int}}

Generate a random graph with `n` vertices, containing a clique that has at least `lbound`
vertices. The probability of generating an edge between any two vertices is `p`, which is
defaulted to `0.1`. The default random number generator is used if not specified. The
function returns a tuple which the first element is the adjacency matrix of containing a
clique specified in the second element.

# Examples
```julia-repl
julia> using Random

julia> (A, c) = generate(10, 4, rng=Xoshiro("Waylon"))
(Bool[0 0 … 0 0; 0 0 … 1 0; … ; 0 1 … 0 1; 0 0 … 1 0], [2, 5, 6, 9])

julia> largestcliques(A)
Main.Cliques.MaximalCliques(:BKPivoting, false, 4, [[2, 5, 6, 9]])
```
"""
function generate(
    n::Int,
    lbound::Int,
    p::Float64=0.1;
    rng::Random.AbstractRNG=Random.default_rng()
)::Tuple{BitMatrix,Vector{Int}}
    A = falses(n, n) # adjacency matrix
    candidates = collect(1:n) # candidates of nodes
    clique = sort!([popat!(candidates, rand(rng, eachindex(candidates))) for _ in 1:lbound]) # random clique
    @inbounds A[clique, clique] .= true # construct clique
    randg = rand(rng, n^2) .< p / 2 # generate edges
    A[:] .|= randg
    A .|= A' # make it symmetric
    foreach(i -> @inbounds(A[i, i] = false), axes(A, 1)) # remove self-loops
    return (A, clique)
end

"""
    Cliques.Schedule

This submodule provides functionality for scheduling the execution of clique-finding
algorithms in a distributed manner.
"""
module Schedule

export scheduler

import ..Cliques
import ..MaximalCliques
import ..complexity, ..largestcliques
import Distributed, StyledStrings

# 1 second in nano-seconds
const second = UInt64(1e9)

# A structure to hold the state of a Bron-Kerbosch pivoting clique search
struct BKCliqueState
    R::BitVector # vertices in the current clique
    P::BitVector # potential vertices to extend the clique
    X::BitVector # vertices excluded
end

# A structure to hold the communication information for a worker
struct Communication
    worker::Int # process id
    controlch::Distributed.Future # control channel to terminate the worker
    updatech::Distributed.RemoteChannel{Channel{NTuple{2,Int}}} # update channel to send updates
    checkint::Int # number of loops to check and update
    minfreq::Float64 # minimal time interval to check and update
end

Communication(worker::Int, checkint::Int=10, minfreq::Float64=1.0) = Communication(
    worker,
    Distributed.Future(Distributed.myid()),
    Distributed.RemoteChannel(() -> Channel{NTuple{2,Int}}(1), Distributed.myid()),
    checkint,
    minfreq
)

# A structure to hold the dynamic scheduling information
struct State
    startedat::Float64 # time started at
    timelimit::Float64 # time limit
    sols::Dict{Symbol,Vector{MaximalCliques}} # solutions
    comms::Dict{Symbol,Vector{Communication}} # communications
    tasks::Dict{Symbol,Vector{Distributed.Future}} # Futures to fetch the results from
end

State(startedat::Float64=time(), timelimit::Float64=55.0) = State(
    startedat,
    timelimit,
    Tuple(
        Dict{Symbol,Vector{t}}(
            a => Vector{t}() for a in (:BKPivoting, :SubgraphRemoval, :Greedy)
        ) for t in (MaximalCliques, Communication, Distributed.Future)
    )...
)

# run `alg` on graph `A` on workers and communication channels given in `comm`
runon(
    A::BitMatrix, comm::Communication, alg::Symbol=:BKPivoting; kwargs...
)::Distributed.Future = Cliques.eval(alg).runon(A, comm; kwargs...)

# Add workers such that total workers is `num`, and load the code on new workers
function needworkers(num::Int)::Vector{Int}
    if Distributed.nworkers() < num
        # add workers
        newprocs = Distributed.addprocs(num + 1 - Distributed.nprocs())
        # load the package everywhere
        Distributed.remotecall_eval(Main, newprocs, :(include($@__FILE__)))
        Distributed.remotecall_eval(Main, newprocs, :(redirect_stdout(devnull)))
        Distributed.remotecall_eval(Main, newprocs, :(redirect_stderr(devnull)))
        return newprocs
    else
        return Vector{Int}()
    end
end

# workers update main process
@inline function update!(info::NTuple{2,Int}, channel::Distributed.RemoteChannel{Channel{NTuple{2,Int}}})::Nothing
    if !isready(channel)
        put!(channel, info) # update channel
    end
    return nothing
end

# Schedule a task to run an algorithm on a worker and return the index of the solution
# placeholder, communication and the task in the vectors. Also a job watcher is returned.
function schedule!(
    A::BitMatrix,
    state::State,
    alg::Symbol,
    pid::Int,
    checkint::Int=10,
    minfreq::Float64=1.0;
    kwargs...
)::Tuple{Int,Task}
    comm = Communication(pid, checkint, minfreq)
    push!(state.comms[alg], comm)
    push!(state.tasks[alg], runon(A, comm, alg; kwargs...))
    push!(state.sols[alg], MaximalCliques(alg, false, 0, Vector{Vector{Int}}()))
    loc = length(state.tasks[alg]) # location of the task
    return (loc, Threads.@spawn fetch(state.tasks[alg][loc]))
end

# Finalise the task by fetching the results, updating the solution, closing update channel,
# and cleaning future placeholers.
function finalise!(
    state::State, alg::Symbol, loc::Int, timeout::Bool=false; kill::Bool=false
)::Union{Tuple{Int,Vector{Vector{Int}}},Tuple{Int,Vararg{Vector{BKCliqueState},2}}}
    fetched = fetch(state.tasks[alg][loc]) # fetch results
    sol = MaximalCliques(fetched, alg, timeout) # update solution
    state.sols[alg][loc] = sol
    # last update
    if isready(state.comms[alg][loc].updatech) # avoid blocking
        take!(state.comms[alg][loc].updatech)
    end
    put!(state.comms[alg][loc].updatech, (sol.omega, length(sol.cliques)))
    finalize(state.tasks[alg][loc]) # finalize task
    finalize(state.comms[alg][loc].controlch) # finalize control channel
    close(state.comms[alg][loc].updatech) # close update channel
    if kill
        Distributed.rmprocs(state.comms[alg][loc].worker, waitfor=0)
    end
    return fetched
end

# Reorder the BK queue based on the greedy solution
reorderbkqueue!(
    bkqueue::Vector{BKCliqueState}, greedysol::MaximalCliques
)::Vector{BKCliqueState} = sort!(
    bkqueue,
    by=c -> @inbounds(
        sum(count((c.R.|c.P)[greedysol.cliques[i]]) for i in eachindex(greedysol.cliques))
    ),
    rev=true
)

# Split the BK queue into `n` subqueues, each containing approximately equal number of elements.
function splitqueue(queue::Vector{BKCliqueState}, n::Int)::Vector{Vector{BKCliqueState}}
    if length(queue) == 0
        return [Vector{BKCliqueState}() for _ in 1:n] # empty queues
    elseif length(queue) < n
        return [i <= n ? [queue[i]] : Vector{BKCliqueState}() for i in 1:n]
    else
        subqlen = round(Int, length(queue) / n) # length of each subqueue
        return [queue[j*subqlen+1:min((j + 1) * subqlen, length(queue))] for j in 0:n-1]
    end
end

# Find and construct the optimal solution from the state.
function findoptimal(state::State, timeout)::MaximalCliques
    # find omega
    omega = maximum(state.sols[alg][s].omega for alg in keys(state.sols) for s in eachindex(state.sols[alg]); init=0)
    # gather all largest cliques
    maxclqs = [
        state.sols[alg][s]
        for alg in keys(state.sols) for s in eachindex(state.sols[alg])
        if state.sols[alg][s].omega == omega
    ]
    # find optimal algorithm
    if any(c -> c.alg == :BKPivoting, maxclqs)
        alg = :BKPivoting
    elseif any(c -> c.alg == :SubgraphRemoval, maxclqs)
        alg = :SubgraphRemoval
    else
        alg = :Greedy
    end
    clqs = [v for c in maxclqs for v in c.cliques]
    unique!(clqs) # remove duplicates
    return MaximalCliques(alg, timeout, omega, clqs)
end

let lastupdate = Dict{Symbol,Vector{Tuple{Int,Int,Bool}}}(
        alg => Vector{Tuple{Int,Int,Bool}}() for alg in (:BKPivoting, :SubgraphRemoval, :Greedy)
    )
    global listtasks
    # List the tasks of a specific algorithm in the state and print their status.
    function listtasks(state::State, alg::Symbol, io::IO)::Int
        line = 0
        runhints = ('◓', '◑', '◒', '◐') # run hints
        omega = maximum(lastupdate[alg][s][1] for alg in keys(lastupdate) for s in eachindex(lastupdate[alg]); init=0)
        if isempty(state.tasks[alg])
            println(io, "  No tasks scheduled.")
            line += 1
        else
            # check update
            for (i, c) in enumerate(state.comms[alg])
                if i > length(lastupdate[alg]) # first run
                    push!(lastupdate[alg], (0, 0, true))
                end
                if lastupdate[alg][i][3] && isready(c.updatech) # update available
                    nextcheck = isopen(c.updatech) # no check if channel is closed
                    lastupdate[alg][i] = (take!(c.updatech)..., nextcheck) # take update
                end
            end
            # print information
            for i in eachindex(state.tasks[alg])
                finalised = state.tasks[alg][i].where == 0
                termntreq = !finalised && isready(state.comms[alg][i].controlch)
                terminated = finalised && state.sols[alg][i].timeout
                if !finalised && !termntreq # running
                    println(
                        io,
                        StyledStrings.styled"  {yellow:$(runhints[mod1(round(Int, time()), 4)]) Running} on worker $(state.comms[alg][i].worker)."
                    )
                    line += 1
                elseif finalised && !terminated # solution found
                    println(
                        io,
                        StyledStrings.styled"  {green:✓ Solution found} on worker $(state.comms[alg][i].worker)."
                    )
                    line += 1
                elseif terminated # terminated
                    println(
                        io,
                        StyledStrings.styled"  {red:✗ Terminated} on worker $(state.comms[alg][i].worker)."
                    )
                    line += 1
                elseif termntreq # termination requested
                    println(
                        io,
                        StyledStrings.styled"  {magenta:⏱ Termination requested} on worker $(state.comms[alg][i].worker)."
                    )
                    line += 1
                else # unknown state
                    println(io, "  ? Unknown state of task $(i) on worker $(state.comms[alg][i].worker).")
                    line += 1
                end
                styledomega = lastupdate[alg][i][1] >= omega ?
                              StyledStrings.styled"{bold,cyan:ω = $(lastupdate[alg][i][1])}" :
                              "ω = $(lastupdate[alg][i][1])"
                println(io, "    ", styledomega, "; found: ", lastupdate[alg][i][2])
                line += 1
            end
        end
        return line
    end
end

let line = 0
    global showst
    # Show the current state of the scheduling and the tasks.
    function showst(state::State, path::AbstractString="the CLQ file", io::IO=stderr)::Nothing
        while line > 0 # clear previous line
            print(io, "\033[1A\033[K")
            line -= 1
        end
        println(io, StyledStrings.styled"{yellow:Searching} for the largest cliques in {italic:$path}...")
        line += 1
        tleft = round(Int, state.startedat + state.timelimit - time()) # time left
        color = tleft <= 10 ? :red : :yellow
        println(io, StyledStrings.styled"Time remaining: {$color:$tleft} sec")
        line += 1
        println(io, StyledStrings.styled"{bold:Bron-Kerbosch pivoting:}")
        line += 1
        line += listtasks(state, :BKPivoting, io) # list BK tasks
        println(io, StyledStrings.styled"{bold:Subgraph removal:}")
        line += 1
        line += listtasks(state, :SubgraphRemoval, io) # list subgraph removal tasks
        println(io, StyledStrings.styled"{bold:Greedy:}")
        line += 1
        line += listtasks(state, :Greedy, io) # list greedy tasks
        flush(io) # flush output
    end
end

# Run the scheduling stage 1 (approximately 1 seconds)
# run greedy; schedule BK on 1
function runstage!(
    A::BitMatrix,
    state::State,
    ::Val{1},
    ::Union{Vector{BKCliqueState},Nothing}=nothing,
    timeout::Float64=1.5
)::Vector{BKCliqueState}
    # initialise
    timewatcher = Threads.@spawn wait(Timer(timeout)) # watcher for timeout
    schedule!(A, state, :Greedy, Distributed.myid()) # schedule greedy
    finalise!(state, :Greedy, length(state.sols[:Greedy]), false) # finalise greedy
    bkloc, bkwatcher = schedule!(A, state, :BKPivoting, Distributed.myid(), 100, 1.0) # schedule BK
    # wait for termination
    while !istaskdone(bkwatcher) && !istaskdone(timewatcher)
        sleep(0.1) # TODO if it really will wait
        yield() # switch to other tasks (BK)
    end
    # terminate
    if istaskdone(timewatcher) # time out
        put!(state.comms[:BKPivoting][bkloc].controlch, true) # terminate BK
    end
    clqsts = fetch(bkwatcher) # fetch results
    finalise!(state, :BKPivoting, bkloc, !isempty(clqsts[3])) # finalize BK
    return clqsts[3] # queue of BK cliques to search
end

# Run the scheduling stage 2 (approximately 10 seconds)
# schedule greedy/re-prioritise BK queue; use 2 processes; schedule BK on 3; schedule subgraph removal on 2
function runstage!(
    A::BitMatrix,
    state::State,
    ::Val{2},
    bkqueue::Union{Vector{BKCliqueState},Nothing}=nothing,
    timeout::Float64=10.5
)::Vector{BKCliqueState}
    # initialise
    timewatcher = Threads.@spawn wait(Timer(timeout)) # watcher for timeout
    addprocstask = @async needworkers(max(2, Sys.CPU_THREADS + 1)) # add process # TODO fall back?
    if isnothing(bkqueue) # first run
        greedyloc, _ = schedule!(A, state, :Greedy, Distributed.myid()) # schedule greedy
        isgreedydone = false # greedy done flag
    else # re-prioritise BK queue
        isgreedydone = true # greedy already done
        reorderbkqueue!(bkqueue, state.sols[:Greedy][1]) # use greedy solution to reorder BK queue
    end
    wait(addprocstask) # wait for process to be added
    if isnothing(bkqueue) # first run
        bkloc, bkwatcher = schedule!(A, state, :BKPivoting, 3, 100, 1.0)
    else # re-schedule BK
        bkloc, bkwatcher = schedule!(A, state, :BKPivoting, 3, 100, 1.0; queue=bkqueue)
    end
    subgrmloc, subgrmwathcer = schedule!(A, state, :SubgraphRemoval, 2, 1, 1.0) # schedule subgraph removal
    subgrmdone = false # subgraph removal done flag
    # wait for termination
    while !istaskdone(bkwatcher) && !istaskdone(timewatcher)
        if !subgrmdone && istaskdone(subgrmwathcer) # subgraph removal done
            finalise!(state, :SubgraphRemoval, subgrmloc, false)
            subgrmdone = true # set flag
        end
        if !isgreedydone && isready(state.tasks[:Greedy][greedyloc]) # greedy done
            finalise!(state, :Greedy, greedyloc, false)
            isgreedydone = true # set flag
        end
        sleep(0.1)
        yield() # switch to other tasks
    end
    # terminate
    if istaskdone(timewatcher) # time out
        put!(state.comms[:BKPivoting][bkloc].controlch, true) # terminate BK
    end
    if !isgreedydone # wait for greedy to finish
        finalise!(state, :Greedy, greedyloc, false)
        isgreedydone = true # set flag
    end
    clqsts = fetch(bkwatcher) # fetch results
    finalise!(state, :BKPivoting, bkloc, !isempty(clqsts[3])) # finalize BK
    return clqsts[3] # queue of BK cliques to search
end

# Run the scheduling stage 3 (approximately 55 seconds) (the final stage)
# schedule greedy/re-prioritise BK queue;
# use Sys.CPU_THREADS processes;
# schedule BK on 3~Sys.CPU_THREADS+1;
# schedule subgraph removal on 2 if first run
function runstage!(
    A::BitMatrix,
    state::State,
    ::Val{3},
    bkqueue::Union{Vector{BKCliqueState},Nothing}=nothing,
    timeout::Float64=55.0
)::Vector{BKCliqueState}
    # initialise
    timewatcher = Threads.@spawn wait(Timer(timeout)) # watcher for timeout
    workers = Sys.CPU_THREADS + 1
    addprocstask = @async needworkers(workers) # add process
    if isnothing(bkqueue) # first run
        greedyloc, _ = schedule!(A, state, :Greedy, Distributed.myid()) # schedule greedy
        bkloc, _ = schedule!(A, state, :BKPivoting, 1, 100, 1.0) # schedule BK
        bktimer = Threads.@spawn wait(Timer(1.0)) # runbk first for 1 second
        finalise!(state, :Greedy, greedyloc, false) # finalise greedy
        firstrun = true
        wait(bktimer) # wait for 1 sec
        put!(state.comms[:BKPivoting][bkloc].controlch, true) # terminate BK
        bkstate = finalise!(state, :BKPivoting, bkloc, true) # finalise BK
        bkqueue = reorderbkqueue!(bkstate[3], state.sols[:Greedy][greedyloc]) # use greedy solution to reorder BK queue
    else # re-prioritise BK queue
        firstrun = false # not first run
        bkqueue = reorderbkqueue!(bkqueue, state.sols[:Greedy][1]) # use greedy solution to reorder BK queue
    end
    subqueues = splitqueue(bkqueue, workers - 1)
    bklocs = Vector{Int}(undef, workers - 1) # locations of BK tasks
    bkwatchers = Vector{Task}(undef, workers - 1) # watchers for BK tasks
    wait(addprocstask) # wait for process to be added
    for n in 1:workers-1 # schedule BK tasks
        bklocs[n], bkwatchers[n] = schedule!(A, state, :BKPivoting, n + 2, 100, 1.0; queue=subqueues[n])
    end
    bkdoneflags = falses(workers - 1) # flags for BK tasks done
    if firstrun
        subgrmloc, subgrmwathcer = schedule!(A, state, :SubgraphRemoval, 2, 1, 1.0) # schedule subgraph removal
    end
    subgrmdone = !isopen(state.comms[:SubgraphRemoval][end].updatech) # subgraph removal done flag
    if !subgrmdone
        subgrmloc = length(state.tasks[:SubgraphRemoval]) # location of subgraph removal task
        subgrmwathcer = Threads.@spawn fetch(state.tasks[:SubgraphRemoval][subgrmloc]) # watcher for subgraph removal
    end
    # wait for termination
    while !all(bkdoneflags) && !istaskdone(timewatcher)
        if !subgrmdone && istaskdone(subgrmwathcer) # subgraph removal done
            finalise!(state, :SubgraphRemoval, subgrmloc, false)
            subgrmdone = true # set flag
        end
        for n in 1:workers-1 # check BK tasks
            if !bkdoneflags[n] && istaskdone(bkwatchers[n]) # BK task done
                finalise!(state, :BKPivoting, bklocs[n], false)
                bkdoneflags[n] = true # set flag
            end
        end
        sleep(0.1)
        yield() # switch to other tasks
    end
    # terminate
    runninglocs = Vector{Int}()
    if istaskdone(timewatcher) # time out
        for n in 1:workers-1 # terminate BK tasks
            cc = state.comms[:BKPivoting][bklocs[n]].controlch # control channel
            if cc.where != 0 # if not finalised
                put!(cc, true) # terminate BK
                push!(runninglocs, n) # save locations
            end
        end
    end
    clqstss = fetch.(bkwatchers) # fetch results
    for n in runninglocs # finalise BK tasks
        finalise!(state, :BKPivoting, bklocs[n], !isempty(clqstss[n][3]))
    end
    # merge BK queues)
    return reduce(vcat, clqstss[n][3] for n in 1:workers-1; init = Vector{BKCliqueState}())
end

"""
    scheduler(
        A::BitMatrix,
        timeout::Float64=59.0,
        updatefreq::Float64=0.9,
        path::AbstractString="the CLQ file",
        io::IO=stdout
    )::MaximalCliques

Schedule the execution of clique-finding algorithms on a distributed system. The graph is
represented by the bit matrix `A`. `timeout` specifies the maximum time
allowed for the scheduling process, defaulting to 55 seconds. `updatefreq` specifies the
frequency of updating the given `io`. `path` provide the path of the CLQ file that `A` was
read from, and will be printed in `io`. The default is "the CLQ file". The function returns
the largest cliques found in the graph as a `MaximalCliques` object.

# Examples
```julia-repl
julia> using .Cliques, .Cliques.Schedule

julia> (A, c) = generate(300, 20, 0.6, rng=Xoshiro("Waylon"))
(Bool[0 1 … 1 1; 1 0 … 0 1; … ; 1 0 … 0 1; 1 1 … 1 0], [11, 28, 45, 67, 84, 91, 112, 120, 138, 152, 153, 169, 170, 222, 223, 235, 247, 248, 289, 300])

julia> scheduler(A)
Searching for the largest cliques in the CLQ file...
Time remaining: 32 sec
Bron-Kerbosch pivoting:
  ✗ Terminated on worker 3.
    ω = 3; found: 1
  ✓ Solution found on worker 3.
    ω = 20; found: 1
  ✓ Solution found on worker 4.
    ω = 13; found: 2
  ✓ Solution found on worker 5.
    ω = 12; found: 10
  ✓ Solution found on worker 6.
    ω = 12; found: 1
Subgraph removal:
  ✓ Solution found on worker 2.
    ω = 8; found: 6
Greedy:
  ✓ Solution found on worker 1.
    ω = 3; found: 1

Main.Cliques.MaximalCliques(:BKPivoting, false, 20, [[11, 28, 45, 67, 84, 91, 112, 120, 138, 152, 153, 169, 170, 222, 223, 235, 247, 248, 289, 300]])
```
"""
function scheduler(A::BitMatrix, timeout::Float64=55.0, updatefreq::Float64=0.9, path::AbstractString="the CLQ file", io::IO=stdout)::MaximalCliques
    # initialise
    state = State(time(), timeout)
    complex = complexity(A, :BKPivoting)
    bkqueue::Union{Vector{BKCliqueState},Nothing} = nothing # queue of BK cliques to search
    updatetimer = updatefreq > 0.0 ?
                  Timer((_) -> showst(state, path, io), updatefreq; interval=updatefreq) :
                  Timer(1) # update timer if applicable
    # run algorithms
    if complex == 1.0 # stage 1
        bkqueue = runstage!(A, state, Val(1), bkqueue, 1.5)
        if isempty(bkqueue) # optimal solution found
            complex = -Inf
        else # more search needed
            complex = 10.0 # set complexity to stage 2
        end
    end
    if complex == 10.0 # stage 2
        bkqueue = runstage!(A, state, Val(2), bkqueue, state.startedat + 10.5 - time())
        if isempty(bkqueue) # optimal solution found
            complex = -Inf
        else # more search needed
            complex = Inf # set complexity to stage 3
        end
    end
    if complex == Inf # stage 3
        bkqueue = runstage!(A, state, Val(3), bkqueue, state.startedat + timeout - time() - 3.0)
        if isempty(bkqueue) # optimal solution found
            complex = -Inf
        end
    end
    # finalise
    if !isempty(state.comms[:SubgraphRemoval]) && isopen(state.comms[:SubgraphRemoval][1].updatech)
        # subgraph removal is still running
        put!(state.comms[:SubgraphRemoval][1].controlch, true) # terminate subgraph removal
        Threads.@spawn finalise!(state, :SubgraphRemoval, 1, true)
        patience = Threads.@spawn wait(Timer(3)) # wait for at most 3 seconds
        while !istaskdone(patience) && isopen(state.comms[:SubgraphRemoval][1].updatech)
            sleep(0.05)
            yield() # switch to other tasks
        end
    end
    if Distributed.nprocs() > 1 # remove processes
        Distributed.interrupt(2:Distributed.nprocs()) # interrupt all workers
        try
            Distributed.rmprocs(2:Distributed.nprocs(), waitfor=2)
        catch
            Distributed.rmprocs(2:Distributed.nprocs(), waitfor=0)
        end
    end
    close(updatetimer) # close update timer
    showst(state, path, io) # update last time
    if updatefreq > 0.0
        print(io, '\n') # add a empty line if applicable
    end
    larclq = findoptimal(state, !isempty(bkqueue)) # find optimal solution
    return larclq # return largest cliques found
end

# Macro to initialise the distributed communication controlers in the algorithm functions
macro init_dist!(comm::Symbol=:comm)
    return esc(
        quote
            distributed = !isnothing($comm) # is in distributed mode
            if distributed # initialise communication controlers
                checkint = $comm.checkint # number of loops to check for termination
                loop = 0 # loop counter
                updt = $comm.minfreq > 0.0 # update remote channel
                minfreq = UInt64($comm.minfreq * second)
                lastchecked = time_ns()
            end
        end
    )
end

# Macro to communicate the current state of the algorithm to the main process
macro communicate!(info_expr::Expr=:(0, 0))
    return esc(
        quote
            if distributed # communicate
                loop += 1
                loopflag = loop > checkint # check if loop counter exceeded
                if loopflag && time_ns() - lastchecked > minfreq # communicate
                    terminate = isready(comm.controlch) # check if termination requested
                    if updt # update remote channel
                        update!($info_expr, comm.updatech)
                    end
                    lastchecked = time_ns()
                    if Distributed.myid() == 1
                        yield() # in case of blocking
                    end
                end
                if loopflag # reset loop counter
                    loop = 0
                end
            end
        end
    )
end

end # module Schedule


# Bron-Kerbosch pivoting clique search algorithm
module BKPivoting

import ..MaximalCliques
import ..Schedule: second, Communication, BKCliqueState, update!, @init_dist!, @communicate!
import Distributed

@inline Base.size(c::BKCliqueState)::Int = count(c.R) # size of the clique
@inline Base.deepcopy(c::BKCliqueState)::BKCliqueState = BKCliqueState(copy(c.R), copy(c.P), copy(c.X))

MaximalCliques(bktuple::Tuple{Int,Vararg{Vector{BKCliqueState},2}}, ::Symbol=:BKPivoting, timeout::Bool=false) =
    MaximalCliques((bktuple[1], map(vertices, bktuple[2])), :BKPivoting, timeout)

# readable representation of the clique
@inline vertices(c::BKCliqueState)::Vector{Int} = findall(c.R)

# choose next pivot vertex
@inline function choosepivot(clq::BKCliqueState)::Int
    for pivot in eachindex(clq.P)
        @inbounds if clq.P[pivot] || clq.X[pivot]
            return pivot
        end
    end
end

# Bron-Kerbosch pivoting algorithm to find maximal cliques in a bit matrix `A`.
# Reference:
#   Johnston, H. C. (1976), "Cliques of a graph—variations on the Bron–Kerbosch algorithm",
#   International Journal of Parallel Programming, 5 (3): 209–238, doi:10.1007/BF00991836,
#   S2CID 29799145.
function bkpivot!(
    A::BitMatrix;
    queue::Vector{BKCliqueState}=[
        BKCliqueState(falses(size(A, 1)), trues(size(A, 1)), falses(size(A, 1)))
    ],
    comm::Union{Communication,Nothing}=nothing
)::Tuple{Int,Vararg{Vector{BKCliqueState},2}}
    # Initialise
    cliques = Vector{BKCliqueState}() # cliques found
    omega::Int = 0 # largest size of clique found
    candidates = BitVector(falses(size(A, 1))) # candidates to grow clique
    neighbours = BitVector(falses(size(A, 1))) # neighbours of a vertex
    terminate::Bool = false # termination flag
    @init_dist! comm
    # loop and grow
    while !isempty(queue) && !terminate
        clq = pop!(queue)
        if !any(clq.P) && !any(clq.X) # maximal clique found
            clqsize = size(clq) # size of the clique
            if clqsize >= omega # a largest clique found
                if clqsize > omega # found a larger clique
                    empty!(cliques) # reset cliques
                    omega = clqsize
                end
                if length(cliques) < 1000 # limit number of cliques
                    push!(cliques, clq)
                end
            end
        else # clique can grow
            pivot = choosepivot(clq)
            @inbounds @. candidates = clq.P & !A[:, pivot] # candidates to grow clique
            @inbounds for v in findall(candidates)
                # grow clique
                grown = deepcopy(clq)
                grown.R[v] = true
                neighbours .= A[:, v]
                grown.P .&= neighbours
                grown.X .&= neighbours
                push!(queue, grown)
                # update P and X
                clq.P[v] = false
                clq.X[v] = true
            end
        end
        @communicate! (omega, length(cliques))
    end
    return (omega, cliques, queue)
end

largestcliques(A::BitMatrix)::MaximalCliques = MaximalCliques(bkpivot!(A))

runon(
    A::BitMatrix,
    comm::Communication;
    queue::Vector{BKCliqueState}=[
        BKCliqueState(falses(size(A, 1)), trues(size(A, 1)), falses(size(A, 1)))
    ]
)::Distributed.Future = Distributed.@spawnat comm.worker bkpivot!(A; queue=queue, comm=comm)

function complexity(A::BitMatrix, factor::Float64=1.0, order::Int=0)::Float64
    n = size(A, 1) # number of vertices
    p = count(A) / n^2 # density of the graph
    if p < -n / 380 + 5 / 4
        return 1.0 * factor * 10.0^order
    elseif p < -n * 3 / 2200 + 25 / 22
        return 10.0 * factor * 10.0^order
    else
        return Inf
    end
end

end # module BKPivoting


# Subgraph removal clique search algorithm
module SubgraphRemoval

import ..MaximalCliques
import ..Schedule: second, Communication, update!, @init_dist!, @communicate!
import Distributed

# Emulate the function calling frame to flatten the recursion.
struct RamseyFrame
    caller::Union{RamseyFrame,Nothing}
    state::Ref{Int8}
    v::Int
    neighbours::BitVector
    NGv::BitMatrix
    CI::NTuple{2,NTuple{2,BitVector}}

    @inline function RamseyFrame(
        caller::Union{RamseyFrame,Nothing}, subgraph::BitMatrix
    )::Union{RamseyFrame,Nothing}
        v = findlast(any, eachrow(subgraph)) # vertex to process
        if isnothing(v) # no vertices left
            return nothing
        end
        @inbounds neighbours = subgraph[:, v] # neighbours of the vertex
        return new(
            caller,
            Ref(Int8(0)),
            v,
            neighbours,
            subgraph,
            Tuple(Tuple(falses(size(subgraph, 1)) for _ in 1:2) for _ in 1:2)
        )
    end
end

# Find the subgraph induced by the neighbours of a vertex.
@inline function neibsubgraph(A::BitMatrix, neighbours::BitVector)::BitMatrix
    subgraph = falses(size(A)) # subgraph to process
    @inbounds subgraph[neighbours, neighbours] .= @view A[neighbours, neighbours]
    return subgraph
end

# Find the complement of the subgraph induced by the neighbours of a vertex.
@inline function neibsubgcompl(NGv::BitMatrix, neighbours::BitVector)::BitMatrix
    compl = falses(size(NGv)) # complement of the neighbours
    nonneigb = .!neighbours # non-neighbours of the vertex
    @inbounds compl[nonneigb, nonneigb] .= @view NGv[nonneigb, nonneigb]
    foreach(i -> @inbounds(compl[i, i] = false), findall(neighbours)) # remove self-loops
    return compl
end

# Flatten Ramsey recursion algorithm
# Reference:
#   Boppana, R., Halldórsson, M.M. Approximating maximum independent sets by excluding
#   subgraphs. BIT 32, 180–196 (1992). https://doi.org/10.1007/BF01994876
function ramsey(A::BitMatrix)::NTuple{2,BitVector}
    # initialise
    stk = [RamseyFrame(nothing, A)] # stack of function calls
    local C, I
    # recursion body
    while !isempty(stk)
        frame = last(stk)
        if frame.state[] == Int8(0) # function entry
            nextcall = RamseyFrame(frame, neibsubgraph(frame.NGv, frame.neighbours)) # Ramsey(N(v))
            if !isnothing(nextcall)
                push!(stk, nextcall)
            end
            frame.state[] = Int8(1)
        elseif frame.state[] == Int8(1) # compute ramsey(N̄(v))
            nextcall = RamseyFrame(frame, neibsubgcompl(frame.NGv, frame.neighbours)) # Ramsey(N̄(v))
            if !isnothing(nextcall)
                push!(stk, nextcall)
            end
            frame.state[] = Int8(2)
        elseif frame.state[] == Int8(2) # return
            frame.CI[1][1][frame.v] = true # C1 + v
            frame.CI[2][2][frame.v] = true # I2 + v
            C = argmax(count, (frame.CI[1][1], frame.CI[2][1])) # max(C1, C2)
            I = argmax(count, (frame.CI[1][2], frame.CI[2][2])) # max(I1, I2)
            pop!(stk) # return to caller
            if !isnothing(frame.caller) # not the root frame
                frame.caller.CI[frame.caller.state[]][1] .= C
                frame.caller.CI[frame.caller.state[]][2] .= I
            end
        end
    end
    return (C, I)
end

# Find the maximum independent set by eliminating subgraphs
# Reference:
#   Boppana, R., Halldórsson, M.M. Approximating maximum independent sets by excluding
#   subgraphs. BIT 32, 180–196 (1992). https://doi.org/10.1007/BF01994876
function maxindepset(A::BitMatrix; comm::Union{Communication,Nothing}=nothing)::Tuple{Int,Vector{Vector{Int}}}
    A = copy(A) # protect input
    maxIs = Vector{Vector{Int}}() # maximum independent sets
    omega::Int = 0 # maximum size
    terminate::Bool = false
    @init_dist! comm
    while any(A) && !terminate
        C, I = ramsey(A)
        @inbounds A[:, C] .= false # A = A \ C
        @inbounds A[C, :] .= false
        sizeI = count(I) # size of independent set
        if sizeI > omega # found a larger independent set
            empty!(maxIs) # reset maximum independent sets
            omega = sizeI
            push!(maxIs, findall(I))
        elseif sizeI == omega # found another maximum independent set
            push!(maxIs, findall(I))
        end
        @communicate! (omega, length(maxIs))
    end
    return (omega, maxIs)
end

# Find the largest cliques in a graph by removing subgraphs on the complement of the graph
function largestcliques(A::BitMatrix)::MaximalCliques
    Ac = .!A # complement of the graph
    foreach(i -> @inbounds(Ac[i, i] = false), axes(Ac, 1)) # remove self-loops
    clqs = maxindepset(Ac)
    return MaximalCliques(:SubgraphRemoval, false, clqs[1], clqs[2])
end

function runon(A::BitMatrix, comm::Communication)::Distributed.Future
    Ac = .!A # complement of the graph
    foreach(i -> @inbounds(Ac[i, i] = false), axes(Ac, 1)) # remove self-loops
    return Distributed.@spawnat comm.worker maxindepset(Ac; comm=comm)
end

function complexity(A::BitMatrix, factor::Float64=3.0, order::Int=0)::Float64
    n = size(A, 1) # number of vertices
    return count(A) / n^2 * n / log(n)^2 * factor * 10.0^order
end

end # module SubgraphRemoval


# Greedy clique search algorithm
module Greedy

import ..MaximalCliques
import ..Schedule.Communication
import Distributed

# Greedy algorithm to find the largest clique in a bit matrix `A`. At each step, it adds the
# vertex with the largest degree among the vertices that are connected to the current
# clique. The algorithm will try this procedure on all vertices.
function greedy(A::BitMatrix)::Tuple{Int,Vector{Vector{Int}}}
    # initialise
    degs = vec(count(A, dims=1)) # degree of each vertex
    order = sortperm(degs, rev=true) # order vertices by degree
    omega = 0 # maximum size of clique
    cliques = Vector{Vector{Int}}() # cliques found
    # start greedy algorithm on each vertex
    for s in order
        clq = falses(size(A, 1)) # current clique
        clq[s] = true
        @inbounds neighbours = A[:, s] # shared neighbours
        while any(neighbours) # there are vertices to add
            vo = findfirst(n -> @inbounds(neighbours[n]), order) # index of largest degree neighbour in sorted order
            v = order[vo] # next vertex to add
            @inbounds clq[v] = true # grow clique
            @inbounds neighbours .&= A[:, v] # update shared neighbours
        end
        sizeclq = count(clq) # size of the clique
        if sizeclq > omega # found a larger clique
            empty!(cliques) # reset cliques
            omega = sizeclq
            push!(cliques, findall(clq)) # store the clique
        elseif sizeclq == omega # found another maximum clique
            clqint = findall(clq) # convert to indices
            if !in(clqint, cliques) && length(cliques) < 1000 # check if clique is not already stored; limit size
                push!(cliques, clqint) # store the clique
            end
        end
    end
    return (omega, cliques)
end

largestcliques(A::BitMatrix)::MaximalCliques = MaximalCliques(greedy(A), :Greedy, false)

runon(A::BitMatrix, comm::Communication)::Distributed.Future = Distributed.@spawnat comm.worker greedy(A)

function complexity(A::BitMatrix, factor::Float64=1.0, order::Int=-10)::Float64
    n = size(A, 1) # number of vertices
    return count(A) / n^2 * n^3 * factor * 10.0^order
end

end # module Greedy


"""
    Cliques.CliquesIO

This module provides functions to read a CLQ file and report the largest cliques found in a
graph.
"""
module CliquesIO

export readclq, report_largest
import ..MaximalCliques

"""
    readclq(path::AbstractString)::BitMatrix

Read a CLQ file from the given `path` and return the graph as a `BitMatrix`.

# Examples
```julia-repl
# Examples
```julia-repl
julia> write(
           "./test.clq",
           \"""
           c Mycielsky-3 graph;
           c |V|=5, omega(M3)=3, Chi(M3)=3
           c
           p edge 5 5
           e 1 2
           e 1 3
           e 1 4
           e 2 3
           e 3 5
           e 4 5
           \"""
       )
102

julia> A = readclq("./test.clq") # read the CLQ file
5×5 BitMatrix:
 0  1  1  1  0
 1  0  1  0  0
 1  1  0  0  1
 1  0  0  0  1
 0  0  1  1  0
```
"""
function readclq(path::AbstractString)::BitMatrix
    lines = filter(startswith('e'), readlines(path)) # edge lines
    elist = zeros(Int, length(lines), 2)
    # read edge list
    for (i, line) in enumerate(lines)
        parts = split(line)
        elist[i, 1] = parse(Int, parts[2])
        elist[i, 2] = parse(Int, parts[3])
    end
    if any(isequal(0), elist)
        @warn "Edge list contains 0-based indexing, converting to 1-based indexing."
        elist .+= 1 # convert to 1-based indexing
    end
    n = maximum(elist) # number of vertices
    A = falses(n, n)
    for l in axes(elist, 1)
        A[elist[l, 1], elist[l, 2]] = true
        A[elist[l, 2], elist[l, 1]] = true
    end
    return A
end

"""
    report_largest(
        maxclqs::MaximalCliques,
        datapath::AbstractString="the CLQ file";
        showfirst::Union{Int,Symbol}=10,
        io::IO=stdout
    )::Nothing

Report the largest cliques to structure `maxclqs` found in the graph read from the file
`datapath` to `io`. The `showfirst` argument specifies how many cliques to show in the
report. The default `datapath` is "the CLQ file". First 10 cliques are shown by default. The
function reports to `stdout` by default.

# Examples
```julia-repl
julia> A = BitMatrix([
           0  1  1  1  0
           1  0  1  0  0
           1  1  0  0  1
           1  0  0  0  1
           0  0  1  1  0
       ])

julia> lclq = largestcliques(A) # find the largest cliques
Main.Cliques.MaximalCliques(:BKPivoting, false, 3, [[1, 2, 3]])

julia> report_largest(lclq)
Largest cliques found in the CLQ file:
  By: Bron-Kerbosch pivoting (OPTIMAL SOLUTION)
  Size: ω = 3
  Number of cliques found: 1
  Vertices:
    [1, 2, 3]
```
"""
function report_largest(
    maxclqs::MaximalCliques,
    datapath::AbstractString="the CLQ file";
    showfirst::Union{Int,Symbol}=10,
    io::IO=stdout
)::Nothing
    println(io, "Largest cliques found in ", datapath, ":")
    print(io, "  By: ")
    timeouthint = "(time out; best solution so far)"
    if maxclqs.alg === :BKPivoting
        hint = maxclqs.timeout ? timeouthint : "(OPTIMAL SOLUTION)"
        println(io, "Bron-Kerbosch pivoting ", hint)
    elseif maxclqs.alg === :SubgraphRemoval
        hint = maxclqs.timeout ? timeouthint : "(approximate solution)"
        println(io, "Subgraph removal ", hint)
    elseif maxclqs.alg === :Greedy
        hint = maxclqs.timeout ? timeouthint : "(approximate guess)"
        println(io, "Greedy heuristic ", hint)
    else
        hint = maxclqs.timeout ? timeouthint : "(unknown algorithm)"
        println(io, maxclqs.alg, ' ', hint)
    end
    println(io, "  Size: ω = ", maxclqs.omega)
    println(io, "  Number of cliques found: ", length(maxclqs.cliques))
    println(io, "  Vertices:")
    if showfirst == :all
        showfirst = length(maxclqs.cliques)
    end
    for clq in first(maxclqs.cliques, showfirst)
        println(io, "    ", clq)
    end
    if length(maxclqs.cliques) > showfirst
        println(io, "    ... and ", length(maxclqs.cliques) - showfirst, " more ...")
    end
end

end # module CliquesIO


end # module Cliques
