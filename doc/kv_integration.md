# K/V Database Integration

This guide explains how to persist FAISS indexes in key-value databases like RocksDB, LevelDB, or ETS.

## Overview

FAISS indexes can be serialized to binary format using `faiss:serialize/1` and restored with `faiss:deserialize/1`. This enables:

- Persistent storage of trained indexes
- Index sharing between nodes
- Backup and recovery
- Caching in ETS or other in-memory stores

## Basic Serialization

```erlang
%% Create and populate an index
{ok, Index} = faiss:new(128),
ok = faiss:add(Index, Vectors),

%% Serialize to binary
{ok, Binary} = faiss:serialize(Index),

%% Binary can be stored anywhere
byte_size(Binary).  %% Size depends on index type and data
```

## RocksDB Integration

### Storing Indexes

```erlang
-module(vector_store).
-export([open/1, close/1, save_index/3, load_index/2, delete_index/2]).

open(Path) ->
    rocksdb:open(Path, [{create_if_missing, true}]).

close(Db) ->
    rocksdb:close(Db).

save_index(Db, Name, Index) ->
    {ok, Binary} = faiss:serialize(Index),
    rocksdb:put(Db, Name, Binary, []).

load_index(Db, Name) ->
    case rocksdb:get(Db, Name, []) of
        {ok, Binary} -> faiss:deserialize(Binary);
        not_found -> {error, not_found}
    end.

delete_index(Db, Name) ->
    rocksdb:delete(Db, Name, []).
```

### Usage Example

```erlang
%% Open database
{ok, Db} = vector_store:open("/tmp/vectors"),

%% Create and train an index
{ok, Index} = faiss:index_factory(128, <<"HNSW32">>),
ok = faiss:add(Index, TrainingVectors),

%% Save to RocksDB
ok = vector_store:save_index(Db, <<"embeddings_v1">>, Index),

%% Later, load it back
{ok, LoadedIndex} = vector_store:load_index(Db, <<"embeddings_v1">>),

%% Verify it works
{ok, Distances, Labels} = faiss:search(LoadedIndex, Query, 10),

%% Clean up
ok = faiss:close(Index),
ok = faiss:close(LoadedIndex),
ok = vector_store:close(Db).
```

## ETS Integration

For in-memory caching with persistence:

```erlang
-module(index_cache).
-export([init/0, cache/2, get/1, persist/1, restore/1]).

init() ->
    ets:new(faiss_cache, [named_table, public, {read_concurrency, true}]).

cache(Name, Index) ->
    {ok, Binary} = faiss:serialize(Index),
    ets:insert(faiss_cache, {Name, Binary}),
    ok.

get(Name) ->
    case ets:lookup(faiss_cache, Name) of
        [{Name, Binary}] -> faiss:deserialize(Binary);
        [] -> {error, not_found}
    end.

%% Persist entire cache to file
persist(Filename) ->
    ets:tab2file(faiss_cache, Filename).

%% Restore cache from file
restore(Filename) ->
    ets:file2tab(Filename).
```

## Multi-Index Architecture

For large-scale applications, partition vectors across multiple indexes:

```erlang
-module(sharded_index).
-export([new/2, add/3, search/3]).

-record(sharded, {
    dimension :: pos_integer(),
    shards :: #{integer() => binary()},  %% shard_id => serialized index
    num_shards :: pos_integer()
}).

new(Dimension, NumShards) ->
    Shards = maps:from_list([
        begin
            {ok, Index} = faiss:new(Dimension),
            {ok, Binary} = faiss:serialize(Index),
            ok = faiss:close(Index),
            {I, Binary}
        end || I <- lists:seq(0, NumShards - 1)
    ]),
    #sharded{dimension = Dimension, shards = Shards, num_shards = NumShards}.

%% Route vector to shard based on ID
add(#sharded{shards = Shards, num_shards = N} = State, VectorId, Vector) ->
    ShardId = VectorId rem N,
    Binary = maps:get(ShardId, Shards),
    {ok, Index} = faiss:deserialize(Binary),
    ok = faiss:add(Index, Vector),
    {ok, NewBinary} = faiss:serialize(Index),
    ok = faiss:close(Index),
    State#sharded{shards = Shards#{ShardId => NewBinary}}.

%% Search all shards and merge results
search(#sharded{shards = Shards}, Query, K) ->
    Results = maps:fold(fun(_ShardId, Binary, Acc) ->
        {ok, Index} = faiss:deserialize(Binary),
        {ok, Dist, Lab} = faiss:search(Index, Query, K),
        ok = faiss:close(Index),
        [{Dist, Lab} | Acc]
    end, [], Shards),
    merge_results(Results, K).

merge_results(Results, K) ->
    %% Flatten and sort by distance, take top K
    %% Implementation left as exercise
    Results.
```

## Versioning Indexes

Track index versions for safe updates:

```erlang
-module(versioned_index).
-export([save/3, load_latest/2, list_versions/2]).

%% Key format: {index_name, version}
save(Db, Name, Index) ->
    Version = erlang:system_time(millisecond),
    Key = term_to_binary({Name, Version}),
    {ok, Binary} = faiss:serialize(Index),
    ok = rocksdb:put(Db, Key, Binary, []),
    {ok, Version}.

load_latest(Db, Name) ->
    %% Find latest version
    Prefix = term_to_binary({Name, 0}),
    case find_latest_version(Db, Name) of
        {ok, Version} ->
            Key = term_to_binary({Name, Version}),
            {ok, Binary} = rocksdb:get(Db, Key, []),
            {ok, Index} = faiss:deserialize(Binary),
            {ok, Index, Version};
        error ->
            {error, not_found}
    end.

list_versions(Db, Name) ->
    %% Return all versions for an index name
    %% Implementation depends on RocksDB iterator usage
    [].
```

## Best Practices

### 1. Serialize After Training

For IVF indexes, serialize after training to avoid retraining:

```erlang
%% Train once
{ok, Index} = faiss:index_factory(128, <<"IVF1024,Flat">>),
ok = faiss:train(Index, TrainingData),

%% Save trained (empty) index as template
{ok, Template} = faiss:serialize(Index),
ok = rocksdb:put(Db, <<"ivf_template">>, Template, []),

%% Use template for new indexes
{ok, TemplateBin} = rocksdb:get(Db, <<"ivf_template">>, []),
{ok, NewIndex} = faiss:deserialize(TemplateBin),
ok = faiss:add(NewIndex, NewVectors).
```

### 2. Batch Updates

Avoid serializing after every add:

```erlang
%% Bad: serialize after each add
lists:foreach(fun(Vec) ->
    ok = faiss:add(Index, Vec),
    {ok, Bin} = faiss:serialize(Index),
    rocksdb:put(Db, Key, Bin, [])
end, Vectors).

%% Good: batch adds, serialize once
ok = faiss:add(Index, AllVectorsBinary),
{ok, Bin} = faiss:serialize(Index),
rocksdb:put(Db, Key, Bin, []).
```

### 3. Use Compression

For large indexes, compress before storing:

```erlang
save_compressed(Db, Key, Index) ->
    {ok, Binary} = faiss:serialize(Index),
    Compressed = zlib:compress(Binary),
    rocksdb:put(Db, Key, Compressed, []).

load_compressed(Db, Key) ->
    {ok, Compressed} = rocksdb:get(Db, Key, []),
    Binary = zlib:uncompress(Compressed),
    faiss:deserialize(Binary).
```

### 4. Handle Errors

Always handle deserialization errors:

```erlang
load_safe(Db, Key) ->
    case rocksdb:get(Db, Key, []) of
        {ok, Binary} ->
            try
                faiss:deserialize(Binary)
            catch
                error:Reason ->
                    {error, {corrupt_index, Reason}}
            end;
        not_found ->
            {error, not_found}
    end.
```

## Performance Considerations

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| `serialize/1` | O(n) | Proportional to index size |
| `deserialize/1` | O(n) | Proportional to binary size |
| Storage size | ~4 bytes/dim/vector | Flat index, uncompressed |

For very large indexes (millions of vectors), consider:

- Using file-based storage (`faiss:write_index/2`) instead of K/V
- Sharding across multiple indexes
- Using compressed index types (PQ, SQ)
