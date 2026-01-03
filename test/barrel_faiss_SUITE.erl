-module(barrel_faiss_SUITE).

-include_lib("common_test/include/ct.hrl").

-export([all/0, groups/0, init_per_suite/1, end_per_suite/1]).
-export([
    basic_new_test/1,
    index_factory_test/1,
    add_search_test/1,
    serialize_deserialize_test/1,
    file_io_test/1,
    hnsw_test/1
]).

all() ->
    [{group, basic}, {group, search}, {group, persistence}].

groups() ->
    [
        {basic, [sequence], [basic_new_test, index_factory_test]},
        {search, [sequence], [add_search_test, hnsw_test]},
        {persistence, [sequence], [serialize_deserialize_test, file_io_test]}
    ].

init_per_suite(Config) ->
    Config.

end_per_suite(_Config) ->
    ok.

%% Test basic index creation
basic_new_test(_Config) ->
    Dim = 128,
    {ok, Index} = barrel_faiss:new(Dim),
    128 = barrel_faiss:dimension(Index),
    true = barrel_faiss:is_trained(Index),
    0 = barrel_faiss:ntotal(Index),
    ok = barrel_faiss:close(Index),
    ok.

%% Test index_factory
index_factory_test(_Config) ->
    Dim = 64,
    {ok, Flat} = barrel_faiss:index_factory(Dim, <<"Flat">>),
    64 = barrel_faiss:dimension(Flat),
    true = barrel_faiss:is_trained(Flat),
    ok = barrel_faiss:close(Flat),

    {ok, FlatIP} = barrel_faiss:index_factory(Dim, <<"Flat">>, inner_product),
    64 = barrel_faiss:dimension(FlatIP),
    ok = barrel_faiss:close(FlatIP),
    ok.

%% Test add and search
add_search_test(_Config) ->
    Dim = 4,
    K = 3,
    {ok, Index} = barrel_faiss:new(Dim),

    %% Create 5 vectors: [1,1,1,1], [2,2,2,2], [3,3,3,3], [4,4,4,4], [5,5,5,5]
    Vectors = <<
        1.0:32/float-native, 1.0:32/float-native, 1.0:32/float-native, 1.0:32/float-native,
        2.0:32/float-native, 2.0:32/float-native, 2.0:32/float-native, 2.0:32/float-native,
        3.0:32/float-native, 3.0:32/float-native, 3.0:32/float-native, 3.0:32/float-native,
        4.0:32/float-native, 4.0:32/float-native, 4.0:32/float-native, 4.0:32/float-native,
        5.0:32/float-native, 5.0:32/float-native, 5.0:32/float-native, 5.0:32/float-native
    >>,
    ok = barrel_faiss:add(Index, Vectors),
    5 = barrel_faiss:ntotal(Index),

    %% Query: [1.5, 1.5, 1.5, 1.5] - should be closest to [1,1,1,1] and [2,2,2,2]
    Query = <<1.5:32/float-native, 1.5:32/float-native, 1.5:32/float-native, 1.5:32/float-native>>,
    {ok, DistBin, LabelBin} = barrel_faiss:search(Index, Query, K),

    %% Parse results
    Distances = [D || <<D:32/float-native>> <= DistBin],
    Labels = [L || <<L:64/signed-native>> <= LabelBin],

    ct:pal("Distances: ~p~n", [Distances]),
    ct:pal("Labels: ~p~n", [Labels]),

    %% Verify we got K results
    K = length(Distances),
    K = length(Labels),

    %% First result should be label 0 or 1 (closest vectors)
    true = lists:member(hd(Labels), [0, 1]),

    ok = barrel_faiss:close(Index),
    ok.

%% Test serialize/deserialize
serialize_deserialize_test(_Config) ->
    Dim = 8,
    {ok, Index1} = barrel_faiss:new(Dim),

    %% Add some vectors
    Vectors = <<
        1.0:32/float-native, 2.0:32/float-native, 3.0:32/float-native, 4.0:32/float-native,
        5.0:32/float-native, 6.0:32/float-native, 7.0:32/float-native, 8.0:32/float-native,
        9.0:32/float-native, 10.0:32/float-native, 11.0:32/float-native, 12.0:32/float-native,
        13.0:32/float-native, 14.0:32/float-native, 15.0:32/float-native, 16.0:32/float-native
    >>,
    ok = barrel_faiss:add(Index1, Vectors),
    2 = barrel_faiss:ntotal(Index1),

    %% Serialize
    {ok, Binary} = barrel_faiss:serialize(Index1),
    true = is_binary(Binary),
    true = byte_size(Binary) > 0,

    %% Deserialize
    {ok, Index2} = barrel_faiss:deserialize(Binary),
    8 = barrel_faiss:dimension(Index2),
    2 = barrel_faiss:ntotal(Index2),

    %% Search should give same results
    Query = <<1.0:32/float-native, 2.0:32/float-native, 3.0:32/float-native, 4.0:32/float-native,
              5.0:32/float-native, 6.0:32/float-native, 7.0:32/float-native, 8.0:32/float-native>>,
    {ok, Dist1, Lab1} = barrel_faiss:search(Index1, Query, 1),
    {ok, Dist2, Lab2} = barrel_faiss:search(Index2, Query, 1),
    Dist1 = Dist2,
    Lab1 = Lab2,

    ok = barrel_faiss:close(Index1),
    ok = barrel_faiss:close(Index2),
    ok.

%% Test file I/O
file_io_test(Config) ->
    Dim = 16,
    PrivDir = ?config(priv_dir, Config),
    Path = list_to_binary(filename:join(PrivDir, "test_index.faiss")),

    {ok, Index1} = barrel_faiss:new(Dim),
    Vectors = random_vectors(10, Dim),
    ok = barrel_faiss:add(Index1, Vectors),
    10 = barrel_faiss:ntotal(Index1),

    %% Write to file
    ok = barrel_faiss:write_index(Index1, Path),

    %% Read from file
    {ok, Index2} = barrel_faiss:read_index(Path),
    16 = barrel_faiss:dimension(Index2),
    10 = barrel_faiss:ntotal(Index2),

    ok = barrel_faiss:close(Index1),
    ok = barrel_faiss:close(Index2),
    ok.

%% Test HNSW index
hnsw_test(_Config) ->
    Dim = 32,
    {ok, Index} = barrel_faiss:index_factory(Dim, <<"HNSW32">>),
    32 = barrel_faiss:dimension(Index),
    true = barrel_faiss:is_trained(Index),

    %% Add vectors
    Vectors = random_vectors(100, Dim),
    ok = barrel_faiss:add(Index, Vectors),
    100 = barrel_faiss:ntotal(Index),

    %% Search
    Query = random_vectors(1, Dim),
    {ok, _Dist, Labels} = barrel_faiss:search(Index, Query, 5),
    5 = byte_size(Labels) div 8,

    ok = barrel_faiss:close(Index),
    ok.

%% Helper: generate random vectors
random_vectors(N, Dim) ->
    << <<(rand:uniform()):32/float-native>> || _ <- lists:seq(1, N * Dim) >>.
