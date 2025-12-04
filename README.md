# Cliques.jl

This module provides functionality for finding the largest cliques in a graph using various
algorithms, including Bron-Kerbosch pivoting, subgraph removal, and greedy methods. It also
includes utilities for validating cliques, generating random graphs containing cliques, and
scheduling the execution of these algorithms in a distributed manner.

This package was developed when I was taking a course on Complex Network Modelling and Inference at The University of Adelaide in 2025.
Further development and maintenance is not guaranteed.
Contributions are welcome.

## Installation

First, clone the repository:
```bash
git clone git@github.com:waylonwh/Networks.jl.git
```

Open Julia from within the local directory of the repo via:
```bash
julia --project
```

The first time, you need to install any dependencies:
```julia
julia> using Pkg; Pkg.instantiate()
```

## Usage
```julia-repl
julia> using Cliques, Cliques.CliquesIO, Cliques.Schedule

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

julia> validate(A, lclq)
true

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
