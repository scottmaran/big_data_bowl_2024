import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import json

from sklearn.manifold import TSNE

from importlib import reload

def importOrReload(module_name, *names):
    import sys

    if module_name in sys.modules:
        reload(sys.modules[module_name])
    else:
        __import__(module_name, fromlist=names)

    for name in names:
        globals()[name] = getattr(sys.modules[module_name], name)

''' 
ids: numpy array (23,) (including football id of -1)
pos_embedding_df: df containing columns ['displayName', embedding dimensions, 'nflId', 'position']

returns the 
'''
def return_pos_embeddings(pos_embedding_df : pd.DataFrame, ids : torch.tensor):
    embedding_dim = pos_embedding_df.shape[1]-3
    cols = [str(x) for x in range(embedding_dim)]
    all_embeds = np.zeros((ids.shape[0], ids.shape[1], embedding_dim))

    for i in range(all_embeds.shape[0]):
        frame_ids = ids[i].numpy()
        embeddings_in_frame = pos_embedding_df.set_index("nflId").loc[frame_ids,cols]     # df
        ids_extracted = embeddings_in_frame.index.values
        msk = np.isin(frame_ids, ids_extracted) # may not have all frame ids in position embedding df, leave absent cases as zero
        all_embeds[i,msk,:] = embeddings_in_frame.values
    
    return torch.tensor(all_embeds, dtype=torch.float32)

def return_pos_df(normalize=True):
    # load in positional embeddings dataframe
    try:
        pos_embeddings = pd.read_csv("./create_pos_embedding/embeddings/2022_full_player_embed_df.csv")
    except:
        pos_embeddings = pd.read_csv("../create_pos_embedding/embeddings/2022_full_player_embed_df.csv")
    pos_embeddings.loc[pos_embeddings.nflId==0,'nflId'] = -1.0
    embed_32_dropped = pos_embeddings.drop(columns=['displayName','nflId', 'position'])
    tsne = TSNE(n_components=2)
    node_embeddings_2d = tsne.fit_transform(embed_32_dropped)
    new_embeddings = pos_embeddings.loc[:,['displayName']]
    new_embeddings['0'] = node_embeddings_2d[:,0]
    new_embeddings['1'] = node_embeddings_2d[:,1]
    new_embeddings['nflId'] = pos_embeddings['nflId']
    new_embeddings['position'] = pos_embeddings['position']

    if normalize:
        max = new_embeddings.loc[:,['0', '1']].max(axis=0).values
        min = new_embeddings.loc[:,['0', '1']].min(axis=0).values
        mean = new_embeddings.loc[:,['0', '1']].mean(axis=0).values
        new_embeddings.loc[:,['0','1']] = (new_embeddings.loc[:,['0','1']] - mean)/(max-min)

    return new_embeddings


### 
### Helper Function for data collection
### 
'''
https://www.kaggle.com/code/statsbymichaellopez/nfl-tracking-wrangling-voronoi-and-sonars/notebook
Function that preprocesses the NGS data, single frame
Changes ngs_frame_df "by_reference", i.e. makes changes to pandas dataframe passed in

- offensive team always moving from left to right
- direction & oritentation
    0 degrees: the offensive player is moving completely to his left
    90 degrees: the offensive player is moving straight ahead, towards opponent end zone
    180 degrees: the offensive player is moving completely to his right
    270 degrees: the offensive player is moving backwards, towards his own team's end zone (this is generally bad)

KEY INSIGHT(?) - how to sort. in seq_sorted_data, sort by attacking team first (so football first, then defensive team, then offensive)
'''
def augment_ngs_frame(attacking_team : str, ball_carrier : int, ngs_frame_df : pd.DataFrame):
    ngs_frame_df.loc[:,'attacking_team'] = (ngs_frame_df.club == attacking_team).astype(int)
    ngs_frame_df.loc[:,'football'] = (ngs_frame_df.club == 'football').astype(int)
    ngs_frame_df.loc[:,'ball_carrier'] = (ngs_frame_df.nflId == ball_carrier).astype(int)

    ''' standardize play direction'''
    ToLeft = np.where(ngs_frame_df.playDirection == "left", True, False)

    ngs_frame_df.loc[:,'x_adj'] = np.where(ToLeft, 120-ngs_frame_df.x, ngs_frame_df.x) - 10 # Standardizes X
    ngs_frame_df.loc[:,'y_adj'] = np.where(ToLeft, 160/3-ngs_frame_df.y, ngs_frame_df.y)    # Standardized Y

    Dir_std_1 = np.where(ToLeft & (ngs_frame_df.dir < 90), ngs_frame_df.dir + 360, ngs_frame_df.dir)   # standardize dir
    Dir_std_1 = np.where( (~ToLeft) & (ngs_frame_df.dir > 270), ngs_frame_df.dir - 360, Dir_std_1)
    ngs_frame_df['dir_adj'] = np.where(ToLeft, Dir_std_1 - 180, Dir_std_1)

    o_std_1 = np.where(ToLeft & (ngs_frame_df.o < 90), ngs_frame_df.o + 360, ngs_frame_df.o)   # standardize o
    o_std_1 = np.where( (~ToLeft) & (ngs_frame_df.dir > 270), ngs_frame_df.dir - 360, o_std_1)
    ngs_frame_df['dir_o'] = np.where(ToLeft, o_std_1 - 180, o_std_1)

    ngs_frame_df.loc[ngs_frame_df.football == 1, ['dir_adj', 'dir_o']] = 0
    ngs_frame_df.sort_values(['attacking_team', 'nflId'], ascending=True, inplace=True)   #smallest first, go to largest




''' 
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
'''
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    
''' create full NGS dataframe '''
def create_ngs_df():
    ngs_df = pd.read_csv("./data/tracking_week_1.csv")
    print(f"dataframe shape = {ngs_df.shape}")
    for i in range(2,10):
        ngs_df = pd.concat([ngs_df,pd.read_csv(f"./data/tracking_week_{i}.csv") ])
    return ngs_df