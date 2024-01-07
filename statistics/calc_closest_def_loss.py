import pandas as pd
import numpy as np
import math
import json
import time
import pickle
import sys

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../')
torch.manual_seed(1)

from dataset import SeqDataset

DATA_DIR = "seq_filtered_sorted_data"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device('cpu')
print(f"Using device = {DEVICE}")

TRAIN_SPLIT = 0.6

with open(f'../cleaned_data/{DATA_DIR}/game_play_id.json') as f:
        list_IDS = json.load(f)
# break ids into train-val-test sets
val_percent = int(len(list_IDS)* (TRAIN_SPLIT+((1-TRAIN_SPLIT)/2)) )
train_IDS, val_IDS, test_IDS = np.split(list_IDS, [ int(len(list_IDS)*TRAIN_SPLIT), val_percent ])

val_params = {'batch_size': 2,
        'shuffle': True,
        'num_workers': 0}
validation_set = SeqDataset(val_IDS, data_dir=DATA_DIR)
validation_generator = torch.utils.data.DataLoader(validation_set, **val_params)

# seq : shape (seq_length, 24, 9)
def get_closest_defender(seq):
    ball_and_opps = seq[:,0:12, 3:5]
    dists = torch.cdist(ball_and_opps[:,0:1,:], ball_and_opps, p=2.0)
    dists[:,0,0] = torch.inf 
    closest_defender = dists.argmin(dim=2).squeeze(-1)
    return closest_defender

def process_batch(device, batch, labels, lengths, player_ids):

    local_batch = batch.to(device)
    index_labels = torch.argmax(labels, -1).to(device) # (batch_size)
    local_lengths = lengths.to(device)    # [batch_size, 23, embed_dim]


    batch_seq_loss = 0  # avg avg seq loss (e.g, on expected sequence, average loss)
    val_metrics_dict = {"quarter_output_pred":0,
                   "halfway_output_pred":0,
                   "three_quarter_output_pred":0,
                   "final_output_pred":0,
                   "correct_tackler_identified_w_highest_prob_anytime":0,
                   "correct_tackler_had_highest_average_prob":0,
                   "correct_tackler_average_prob":0}
    
    # for each full sequence
    for i in range(0, local_batch.shape[0]):

        single_target = index_labels[i, local_lengths[i]-1].to(device)                                 # ([]), the correct class
        batch = local_batch[i, :local_lengths[i]-1, :].reshape(-1,24,9).to(device)    # (seq_length, 24, 9)

        closest_def = get_closest_defender(batch).to(device)
        correct_classify = (closest_def == single_target).unsqueeze(1).float()

        #batch_seq_loss += F.nll_loss(torch.log(closest_def), index_labels[i, :local_lengths[i]-1], reduction='mean')
        batch_seq_loss += 0

        # shape ([]), 1 or 0 if classified correctly at that point in time
        val_metrics_dict['quarter_output_pred'] += correct_classify[correct_classify.shape[0]//4].item()
        val_metrics_dict['halfway_output_pred'] += correct_classify[correct_classify.shape[0]//2].item()
        val_metrics_dict['three_quarter_output_pred'] += correct_classify[(correct_classify.shape[0]//4)*3].item()
        val_metrics_dict['final_output_pred'] += correct_classify[-1].item()
        val_metrics_dict['correct_tackler_identified_w_highest_prob_anytime'] += min(correct_classify.sum().item(), 1)

    return batch_seq_loss, val_metrics_dict

print(f"Starting training...")

total_start_time = time.time()

avg_val_loss = 0
val_loss_hist = []
val_metrics_dict = {"quarter_output_pred":0,
                "halfway_output_pred":0,
                "three_quarter_output_pred":0,
                "final_output_pred":0,
                "correct_tackler_identified_w_highest_prob_anytime":0,
                "correct_tackler_had_highest_average_prob":0,
                "correct_tackler_average_prob":0}

gen = iter(validation_generator)
num_val_batches = len(gen)

print(f"num val batches = {num_val_batches}")

for batch_index, (local_batch, local_labels, local_lengths, local_player_ids, local_ids) in enumerate(validation_generator):

    batch_seq_loss, val_batch_metrics_dict = process_batch(DEVICE, local_batch, local_labels, local_lengths, local_player_ids)
    
    #val_loss_hist.append(batch_seq_loss.item())
    #avg_val_loss += batch_seq_loss.item()

    val_metrics_dict['quarter_output_pred'] += val_batch_metrics_dict['quarter_output_pred']
    val_metrics_dict['halfway_output_pred'] += val_batch_metrics_dict['halfway_output_pred']
    val_metrics_dict['three_quarter_output_pred'] += val_batch_metrics_dict['three_quarter_output_pred']
    val_metrics_dict['final_output_pred'] += val_batch_metrics_dict['final_output_pred']
    val_metrics_dict['correct_tackler_identified_w_highest_prob_anytime'] += val_batch_metrics_dict['correct_tackler_identified_w_highest_prob_anytime']
    val_metrics_dict['correct_tackler_had_highest_average_prob'] += val_batch_metrics_dict['correct_tackler_had_highest_average_prob']
    val_metrics_dict['correct_tackler_average_prob'] += val_batch_metrics_dict['correct_tackler_average_prob']

avg_val_loss /= (num_val_batches*validation_generator.batch_size)
val_metrics_dict['quarter_output_pred'] /= (num_val_batches*validation_generator.batch_size)
val_metrics_dict['halfway_output_pred'] /= (num_val_batches*validation_generator.batch_size)
val_metrics_dict['three_quarter_output_pred'] /= (num_val_batches*validation_generator.batch_size)
val_metrics_dict['final_output_pred'] /= (num_val_batches*validation_generator.batch_size)
val_metrics_dict['correct_tackler_identified_w_highest_prob_anytime'] /= (num_val_batches*validation_generator.batch_size)
val_metrics_dict['correct_tackler_had_highest_average_prob'] /= (num_val_batches*validation_generator.batch_size)
val_metrics_dict['correct_tackler_average_prob'] /= (num_val_batches*validation_generator.batch_size)


print(f"val_loss={avg_val_loss}")
print(f"val metrics dict = ")
print(f"{list(val_metrics_dict.keys())}")
print(f"{np.array(list(val_metrics_dict.values())).round(3)}")
print(f"#######################")
    
total_end_time = time.time()
print(f"Finished training 1 epochs in {round((total_end_time - total_start_time)/60, 3)} min")

