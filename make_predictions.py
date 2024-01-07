import pickle
import numpy as np
import pandas as pd
import torch
 
from models import trans_single_frame_12_27
from models import trans_single_frame_13_6_nine

from sklearn.manifold import TSNE

from utils import return_pos_embeddings, return_pos_df

# play_ex_id = '2022091102_86'
def make_prediction(play_ex_id, MODEL_DIR):
    DATA_DIR = "seq_clipped_sorted_data"

    NORM=True

    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device('cpu')
    print(f"Using device = {DEVICE}")

    new_embeddings = return_pos_df()

    model = trans_single_frame_13_6_nine(pos_df=new_embeddings, feature_embed_size=128, dropout=0.2, num_encoder_layers=4, num_att_heads=32)
    model.load_state_dict(torch.load(f"/Users/scottmaran/Code/Python/bdb_2024/saved_models/{MODEL_DIR}/weights/trans_all_players.pt"))
    model.eval()
    model = model.to(DEVICE)

    with open(f"/Users/scottmaran/Code/Python/bdb_2024/saved_models/{MODEL_DIR}/weights/norm_list.pickle", 'rb') as fp:
        norm_list = pickle.load(fp)

    if len(norm_list) != 0:
        training_mean, _, normalization_mask, min_max_diff = norm_list
        training_mean = training_mean.to(DEVICE)
        normalization_mask = normalization_mask.to(DEVICE)
        min_max_diff = min_max_diff.to(DEVICE)

    print(f"ID = {play_ex_id}")
    # need to add batch dim to make it pass thru model
    feature = torch.load(f"/Users/scottmaran/Code/Python/bdb_2024/cleaned_data/{DATA_DIR}/features/{play_ex_id}.pt").unsqueeze(0).to(DEVICE)
    label = torch.load(f"/Users/scottmaran/Code/Python/bdb_2024/cleaned_data/{DATA_DIR}/labels/{play_ex_id}.pt").unsqueeze(0).to(DEVICE)
    context = torch.load(f"/Users/scottmaran/Code/Python/bdb_2024/cleaned_data/{DATA_DIR}/context_vector/{play_ex_id}.pt").unsqueeze(0).to(torch.float32).to(DEVICE)
    player_ids = torch.tensor(np.load(f"/Users/scottmaran/Code/Python/bdb_2024/cleaned_data/{DATA_DIR}/player_ids/{play_ex_id}.npy")).unsqueeze(0).to(torch.float32).to(DEVICE)
    with open(f'/Users/scottmaran/Code/Python/bdb_2024/cleaned_data/{DATA_DIR}/play_info_dict/{play_ex_id}', 'rb') as file:
        play_info_dict = pickle.load(file)

    length = play_info_dict['final_frame_in_play']

    single_frame_feature = feature.squeeze(0).reshape(-1,23,11)[0:length].to(DEVICE)
    single_frame_context = context.squeeze(0)[0:length].to(DEVICE)
    single_frame_label = label.squeeze(0).argmax(-1)[0:length].to(DEVICE)

    if len(norm_list) != 0:
        single_frame_feature = single_frame_feature.reshape(-1,23*11)
        single_frame_feature[:,normalization_mask] = (single_frame_feature[:,normalization_mask] - training_mean[normalization_mask].reshape(1,-1))/torch.where(min_max_diff[:,normalization_mask]==0,1,min_max_diff[:,normalization_mask])
        single_frame_feature = single_frame_feature.reshape(-1,23,11)

    pos_embeds = return_pos_embeddings(new_embeddings, player_ids.cpu())
    pos_embeds = pos_embeds.to(DEVICE)

    with torch.no_grad():
        prediction = model(single_frame_feature[:,:,:9], pos_embeddings=pos_embeds, context_vec=single_frame_context)
    # output is log(prob), so turn to probs
    prediction = torch.exp(prediction)

    torch.save(prediction, f"/Users/scottmaran/Code/Python/bdb_2024/saved_models/{MODEL_DIR}/predictions/{play_ex_id}.pt")

    return prediction