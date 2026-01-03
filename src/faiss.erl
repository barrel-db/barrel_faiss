-module(faiss).

%% Index creation
-export([new/1, new/2, index_factory/2, index_factory/3, close/1]).

%% Index properties
-export([dimension/1, is_trained/1, ntotal/1]).

%% Core operations
-export([add/2, search/3, train/2]).

%% Serialization
-export([serialize/1, deserialize/1]).

%% File I/O
-export([write_index/2, read_index/1]).

%% Types
-export_type([index/0, metric_type/0]).

-opaque index() :: reference().
-type metric_type() :: l2 | inner_product.

-on_load(init/0).

-define(nif_stub, erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]})).

init() ->
    PrivDir = code:priv_dir(faiss),
    ok = erlang:load_nif(filename:join(PrivDir, "faiss"), 0).

%% @doc Create a new flat L2 index with given dimension.
-spec new(Dimension :: pos_integer()) -> {ok, index()} | {error, term()}.
new(Dimension) ->
    new(Dimension, l2).

%% @doc Create a new flat index with given dimension and metric.
-spec new(Dimension :: pos_integer(), Metric :: metric_type()) ->
    {ok, index()} | {error, term()}.
new(_Dimension, _Metric) ->
    ?nif_stub.

%% @doc Create index using factory string.
%% Examples: <<"Flat">>, <<"IVF100,Flat">>, <<"HNSW32">>
-spec index_factory(Dimension :: pos_integer(), Description :: binary()) ->
    {ok, index()} | {error, term()}.
index_factory(Dimension, Description) ->
    index_factory(Dimension, Description, l2).

-spec index_factory(Dimension :: pos_integer(), Description :: binary(),
                    Metric :: metric_type()) ->
    {ok, index()} | {error, term()}.
index_factory(_Dimension, _Description, _Metric) ->
    ?nif_stub.

%% @doc Close and release index resources.
-spec close(Index :: index()) -> ok.
close(_Index) ->
    ?nif_stub.

%% @doc Get the dimension of vectors in the index.
-spec dimension(Index :: index()) -> pos_integer() | {error, term()}.
dimension(_Index) ->
    ?nif_stub.

%% @doc Check if the index is trained.
-spec is_trained(Index :: index()) -> boolean() | {error, term()}.
is_trained(_Index) ->
    ?nif_stub.

%% @doc Get the number of vectors in the index.
-spec ntotal(Index :: index()) -> non_neg_integer() | {error, term()}.
ntotal(_Index) ->
    ?nif_stub.

%% @doc Add vectors to the index.
%% Vectors is a binary of packed float32 values.
%% The binary size must be n * dimension * 4 bytes.
-spec add(Index :: index(), Vectors :: binary()) -> ok | {error, term()}.
add(_Index, _Vectors) ->
    ?nif_stub.

%% @doc Search for k nearest neighbors.
%% Queries is a binary of packed float32 values.
%% Returns {ok, Distances, Labels} where both are binaries:
%% - Distances: nq * k float32 values
%% - Labels: nq * k int64 values
-spec search(Index :: index(), Queries :: binary(), K :: pos_integer()) ->
    {ok, Distances :: binary(), Labels :: binary()} | {error, term()}.
search(_Index, _Queries, _K) ->
    ?nif_stub.

%% @doc Train the index with sample vectors.
%% Required for IVF indexes before adding vectors.
-spec train(Index :: index(), Vectors :: binary()) -> ok | {error, term()}.
train(_Index, _Vectors) ->
    ?nif_stub.

%% @doc Serialize index to binary for K/V storage.
-spec serialize(Index :: index()) -> {ok, binary()} | {error, term()}.
serialize(_Index) ->
    ?nif_stub.

%% @doc Deserialize index from binary.
-spec deserialize(Binary :: binary()) -> {ok, index()} | {error, term()}.
deserialize(_Binary) ->
    ?nif_stub.

%% @doc Write index to file.
-spec write_index(Index :: index(), Path :: binary()) -> ok | {error, term()}.
write_index(_Index, _Path) ->
    ?nif_stub.

%% @doc Read index from file.
-spec read_index(Path :: binary()) -> {ok, index()} | {error, term()}.
read_index(_Path) ->
    ?nif_stub.
