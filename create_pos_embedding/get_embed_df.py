import pandas as pd
import numpy as np
import time

from itertools import combinations
import networkx as nx

import node2vec
from gensim.models import Word2Vec

DIM_NUM=3

def create_graph_from_frame(frame_df:pd.DataFrame, nfl_id_tuples:list=None, nfl_ids:np.array=None):
    all_dists_mat = np.linalg.norm(frame_df.loc[:,['x','y']].values - frame_df.loc[:,['x','y']].values[:,None], axis=-1)
    all_dists_tri = all_dists_mat[np.triu_indices(all_dists_mat.shape[0], k=1)]

    if nfl_id_tuples is None:
        nfl_id_tuples = list(combinations(frame_df.nflId, r=2))

    list_of_weight_dics = [{"weight":dist} for dist in all_dists_tri]   # create bc that's input for nx_graph
    edges = [(nfl_id_tuples[i][0], nfl_id_tuples[i][1], list_of_weight_dics[i]) for i in range(len(nfl_id_tuples))]

    nx_G = nx.Graph()
    if nfl_ids is None:
        nx_G.add_nodes_from(frame_df.nflId)
    else:
        nx_G.add_nodes_from(nfl_ids)
    nx_G.add_edges_from(edges)
    nx_G = nx_G.to_undirected()

    return nx_G

# from node2vec github repo
def learn_embeddings(walks):
	dimensions=DIM_NUM
	window_size=10
	epoch_iters = 1

	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [list(map(str, walk)) for walk in walks]
	model = Word2Vec(walks, vector_size=dimensions, window=window_size, min_count=0, sg=1, workers=1, epochs=epoch_iters)
	#model.save_word2vec_format(output)
	return model

def get_model(nx_G):
    directed=False
    p = 1
    q = 1

    num_walks=10
    walk_length=10

    G = node2vec.Graph(nx_G, directed, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    model = learn_embeddings(walks)

    return model

''' 
id_to_arr_dict : dict we add to
ngs_full_play : pandas dataframe with NGS data
'''
def add_player_embeds_from_play(id_to_arr_dict: dict, ngs_full_play: pd.DataFrame):
    frame_ids = ngs_full_play.frameId.unique()
    final_frame_in_play = int(frame_ids.max())

    all_nfl_ids_in_play = ngs_full_play.nflId.unique()  # array
    assert len(all_nfl_ids_in_play) == 23
    nfl_id_tuples = list(combinations(all_nfl_ids_in_play, r=2))

    for frame_id in range(1, final_frame_in_play+1):

        nx_G = create_graph_from_frame(ngs_full_play.loc[ngs_full_play.frameId == frame_id], nfl_id_tuples, all_nfl_ids_in_play)
        model = get_model(nx_G)
        
        for key,index in model.wv.key_to_index.items():
            # if key exists, gets it and adds to it
            id_to_arr_dict[str(key)] = id_to_arr_dict.get(str(key), np.zeros(DIM_NUM)) + model.wv[str(key)]/final_frame_in_play #do this to average

    return id_to_arr_dict

def main():
    games = pd.read_csv("../data/games.csv")
    players = pd.read_csv("../data/players.csv")
    plays = pd.read_csv("../data/plays.csv")

    ngs_df = pd.read_csv("../data/tracking_week_1.csv")
    print(f"dataframe shape = {ngs_df.shape}")
    for i in range(2,9):
        ngs_df = pd.concat([ngs_df,pd.read_csv(f"../data/tracking_week_{i}.csv") ])

    all_games = ngs_df.gameId.unique()
    print(f"Number of games = {len(all_games)}")

    # dictionary mapping {nflid (str) : 32-dim rep.}
    full_id_to_arr_dict = dict()

    total_start_time = time.time()
    for game_index, game_id in enumerate(all_games):
        all_play_ids = plays.query("gameId == @game_id").playId.values

        for play_id in all_play_ids:
            current_id = f"{str(game_id)}_{str(play_id)}"
            try:
                ngs_full_play = ngs_df.loc[(ngs_df.gameId == game_id) & (ngs_df.playId == play_id)]

                add_player_embeds_from_play(full_id_to_arr_dict, ngs_full_play)
            
            except Exception as e:
                print(f"play {current_id}: Exception {e}")
        
    total_end_time = time.time()
    print(f"Finished {len(all_games)} games in {round((total_end_time - total_start_time)/60, 3)} min")

    # convert dictionary of arrays into dataframe
    player_embed_df = pd.DataFrame.from_dict(full_id_to_arr_dict,orient = 'index')
    player_embed_df.index = player_embed_df.index.astype('float64')
    player_embed_df.index = player_embed_df.index.fillna(0)             # change NaN index to 0 for football

    merged = pd.merge(player_embed_df, players.loc[:,['nflId','position', 'displayName']], how='left', left_index=True, right_on='nflId').reset_index(drop=True)
    merged.loc[merged.nflId == 0, 'displayName'] = 'Football'
    player_df = merged.set_index("displayName")

    #player_df.to_csv("2022_3_dim_full_player_embed_df.csv")

if __name__ == "__main__":
    main()