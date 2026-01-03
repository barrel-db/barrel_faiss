# Getting Started

This guide will help you get up and running with the FAISS Erlang bindings.

## Installation

### Prerequisites

Install FAISS on your system:

```bash
# macOS (Homebrew)
brew install faiss libomp

# macOS (MacPorts)
sudo port install libfaiss

# Debian/Ubuntu
apt install libfaiss-dev libomp-dev libopenblas-dev

# FreeBSD
pkg install faiss openblas
```

### Adding to Your Project

Add `faiss` to your `rebar.config` dependencies:

```erlang
{deps, [
    {faiss, {git, "https://github.com/barrel-db/erlang-faiss.git", {branch, "main"}}}
]}.
```

Then compile:

```bash
rebar3 compile
```

## Quick Start

### Step 1: Create an Index

The simplest way to start is with a flat index for exact nearest neighbor search:

```erlang
%% Create a flat index for 128-dimensional vectors
{ok, Index} = faiss:new(128).
```

### Step 2: Prepare Your Vectors

Vectors must be binaries of packed 32-bit floats in native byte order:

```erlang
%% Helper function to convert a list of floats to binary
vectors_to_binary(Vectors) ->
    << <<V:32/float-native>> || V <- lists:flatten(Vectors) >>.

%% Example: 3 vectors of dimension 4
Vectors = [
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0]
],
VectorsBin = vectors_to_binary(Vectors).
```

### Step 3: Add Vectors to the Index

```erlang
ok = faiss:add(Index, VectorsBin),
3 = faiss:ntotal(Index).  %% Verify 3 vectors added
```

### Step 4: Search for Nearest Neighbors

```erlang
%% Create a query vector
Query = vectors_to_binary([[2.0, 3.0, 4.0, 5.0]]),

%% Search for 2 nearest neighbors
{ok, Distances, Labels} = faiss:search(Index, Query, 2),

%% Parse results
DistanceList = [D || <<D:32/float-native>> <= Distances],
LabelList = [L || <<L:64/signed-native>> <= Labels],

io:format("Nearest neighbors: ~p~n", [LabelList]),
io:format("Distances: ~p~n", [DistanceList]).
```

### Step 5: Clean Up

```erlang
ok = faiss:close(Index).
```

## Complete Example

Here's a complete example you can run in the Erlang shell:

```erlang
%% Helper function
VectorsToBinary = fun(Vecs) ->
    << <<V:32/float-native>> || V <- lists:flatten(Vecs) >>
end,

%% Create index
{ok, Index} = faiss:new(4),

%% Add 5 vectors
Vectors = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.5, 0.5, 0.5, 0.5]
],
ok = faiss:add(Index, VectorsToBinary(Vectors)),

%% Search for vector closest to [0.6, 0.6, 0.4, 0.4]
Query = VectorsToBinary([[0.6, 0.6, 0.4, 0.4]]),
{ok, Distances, Labels} = faiss:search(Index, Query, 3),

%% Parse and display results
Results = lists:zip(
    [L || <<L:64/signed-native>> <= Labels],
    [D || <<D:32/float-native>> <= Distances]
),
io:format("Top 3 results (label, distance): ~p~n", [Results]),

%% Clean up
ok = faiss:close(Index).
```

Expected output:
```
Top 3 results (label, distance): [{4,0.04},{1,0.68},{0,0.68}]
```

Vector 4 `[0.5, 0.5, 0.5, 0.5]` is closest to our query.

## Choosing an Index Type

FAISS offers different index types for different use cases:

| Use Case | Index Type | Factory String |
|----------|-----------|----------------|
| Small dataset, exact results | Flat | `<<"Flat">>` |
| Fast approximate search | HNSW | `<<"HNSW32">>` |
| Large dataset (1M+ vectors) | IVF | `<<"IVF1024,Flat">>` |
| Memory constrained | IVF+PQ | `<<"IVF1024,PQ16">>` |

### Using index_factory

```erlang
%% HNSW - fast approximate search, no training needed
{ok, HnswIndex} = faiss:index_factory(128, <<"HNSW32">>),

%% IVF - requires training first
{ok, IvfIndex} = faiss:index_factory(128, <<"IVF100,Flat">>),
false = faiss:is_trained(IvfIndex),

%% Train with sample data (need ~100 vectors per centroid)
TrainingData = generate_training_data(10000, 128),
ok = faiss:train(IvfIndex, TrainingData),
true = faiss:is_trained(IvfIndex).
```

## Metric Types

FAISS supports two distance metrics:

- **L2 (Euclidean)**: Default, smaller distance = more similar
- **Inner Product**: Use for cosine similarity (normalize vectors first)

```erlang
%% L2 distance (default)
{ok, L2Index} = faiss:new(128, l2),

%% Inner product
{ok, IpIndex} = faiss:new(128, inner_product).
```

## Next Steps

- See [K/V Database Integration](kv_integration.md) for persisting indexes
- Check the `faiss` module documentation for the complete API reference
