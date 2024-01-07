import pandas as pd
import numpy as np
import sys
import os
import json
import time
import pickle

import torch

torch.manual_seed(1)
sys.path.append('./')

from dataset import SeqContextDataset, SingleFrameContextDataset
from utils import return_pos_embeddings, return_pos_df

from models import trans_single_frame_12_27

''' 
Process batch of sequence dataset
'''
def process_batch(stats_dict, model, device, batch, labels, context, lengths, player_ids, norm_list:list=[]):
    model = model.to(device)
    local_batch = batch.to(device)
    index_labels = torch.argmax(labels, -1).to(device) # (batch_size)
    local_lengths = lengths.to(device)
    local_context = context.to(torch.float32).to(device)

    # for each full sequence
    for i in range(0, local_batch.shape[0]):

        # load in position embeddings, set up dimensions
        single_target = index_labels[i, local_lengths[i]-1].to(device)                                 # ([]), the correct class
        batch = local_batch[i, :local_lengths[i]-1, :].reshape(-1,23,11).to(device)    # (seq_length, 23,11)
        context = local_context[i,:local_lengths[i]-1,:].to(device)
        pos_embeddings = return_pos_embeddings(model.pos_df, player_ids[i:i+1])         
        pos_embeddings = pos_embeddings.repeat(local_lengths[i]-1,1,1).to(device)       # (seq_length, 23,11)

        # normalize batch
        if len(norm_list) != 0:
            training_mean, _, normalization_mask, min_max_diff = norm_list
            training_mean = training_mean.to(device)
            normalization_mask = normalization_mask.to(device)
            min_max_diff = min_max_diff.to(device)

            batch = batch.reshape(-1,23*11)
            batch[:,normalization_mask] = (batch[:,normalization_mask] - training_mean[normalization_mask].reshape(1,-1))/torch.where(min_max_diff[:,normalization_mask]==0,1,min_max_diff[:,normalization_mask])
            batch = batch.reshape(-1,23,11)

        # get model outputs
        with torch.no_grad():
            output = model(batch, pos_embeddings, context).to(device)   # (seq_length, 23)

        output = torch.exp(output)  # convert logits to probabilities

        max_tackle_probs_over_seq = output.max(dim=0)[0]    # (23)
        avg_tackle_probs_over_seq = output.mean(dim=0)      # (23)
        true_tackler = labels[i, 0]                         # (23)
        delta_x = (torch.roll(output,shifts=1,dims=0) - output)[1:,:].sum(dim=0)    # (23)

        # print(f"output shape  = {output.shape}")
        # print(f"labels shape = {labels.shape} ")
        # print(f"max_tackle_probs_over_seq = {max_tackle_probs_over_seq.shape}")
        # print(f"avg_tackle_probs_over_seq = {avg_tackle_probs_over_seq.shape}")
        # print(f"true_tackler = {true_tackler.shape}")

        #each_players_dict = [{"max_xT":max_tackle_probs_over_seq[k].item(), "xT":avg_tackle_probs_over_seq[k].item(), "T":true_tackler[k].item()} for k in range(len(player_ids))]   # list of len 23 with size 3 dict
        
        player_stats = torch.zeros((23, 4))
        player_stats[:,0] = max_tackle_probs_over_seq
        player_stats[:,1] = avg_tackle_probs_over_seq
        player_stats[:,2] = true_tackler
        player_stats[:,3] = delta_x

        for index, player_id in enumerate(player_ids[i:i+1].squeeze()):
            stats_dict[player_id.item()] = stats_dict.get(player_id.item(), torch.zeros(player_stats.shape[1])) + player_stats[index]
        
        #stats_dict.update(dict(zip(player_ids, each_players_dict)))


def main():

    # cwd = os.getcwd()
    # print(f"################")
    # print(f"CWD = {cwd}")

    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device('cpu')
    DEVICE = torch.device('cpu')
    print(f"Using device = {DEVICE}")
    DATA_DIR = "seq_clipped_sorted_data"
    MODEL_DIR = f"trans_single_frame_12_27_hw"

    NORM=True

    # load in all IDS
    with open(f'./cleaned_data/{DATA_DIR}/game_play_id.json') as f:
        list_IDS = json.load(f)

    # load in dataloaders - should work on sequences instead of single frames
    gen_params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 0}
    dataset = SeqContextDataset(list_IDS, data_dir=DATA_DIR)
    data_loader = torch.utils.data.DataLoader(dataset, **gen_params)

    # load in positional embeddings dataframe
    new_embeddings = return_pos_df()

    # load in model
    model = trans_single_frame_12_27(pos_df=new_embeddings, feature_embed_size=128, dropout=0.2, num_encoder_layers=4, num_att_heads=32)
    model.load_state_dict(torch.load(f"./saved_models/{MODEL_DIR}/weights/trans_all_players.pt"))
    model.eval()
    if NORM:
        with open(f"./saved_models/{MODEL_DIR}/weights/norm_list.pickle", 'rb') as fp:
            norm_list = pickle.load(fp)
    else:
        norm_list = []

    # create stats dict : dict = {nflId : dict{cum. xTackles, cum. Tackles} }
    stats_dict = dict()

    total_start_time = time.time()
    data_generator = iter(data_loader)
    for batch_index in range(len(data_generator)):

        if batch_index == 1309:
            print(f"at batch_index 1309")

        with torch.no_grad():
            local_batch, local_labels, local_context, local_lengths, local_player_ids, local_ids = next(data_generator)
            process_batch(stats_dict, model, DEVICE, local_batch, local_labels, local_context, local_lengths, local_player_ids, norm_list)
        if batch_index % 500 == 0:
            print(f"finished batch {batch_index} of {len(data_generator)} in {round((time.time() - total_start_time)/60, 3)} min")

    total_end_time = time.time()
    print(f"Finished all seq. in {round((total_end_time - total_start_time)/60, 3)} min")

    # save expected tackles per player
    expected_tackle_df = pd.DataFrame(stats_dict).T
    expected_tackle_df.to_csv(f"statistics/expected_tackle_df_{MODEL_DIR}.csv")

if __name__ == "__main__":
    main()