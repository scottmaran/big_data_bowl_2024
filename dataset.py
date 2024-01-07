'''
Characterizes a dataset for PyTorch
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
'''

import torch
import numpy as np
import pickle

'''
data_dir = [seq_clipped_sorted_data]

seq_filtered_sorted_data : X = [batch_size, max_seq_len, 23*11]
'''
class SeqContextDataset(torch.utils.data.Dataset):
  # takes as input json file
  def __init__(self, list_IDS : list, data_dir : str):
        self.list_IDs = list_IDS
        self.data_dir = data_dir

        self.num_agents=23
        self.feature_size=11
        self.context_length=9

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        X = torch.load(f'/Users/scottmaran/Code/Python/bdb_2024/cleaned_data/{self.data_dir}/features/' + ID + '.pt')
        y = torch.load(f'/Users/scottmaran/Code/Python/bdb_2024/cleaned_data/{self.data_dir}/labels/' + ID + '.pt')
        context_vector = torch.load(f'/Users/scottmaran/Code/Python/bdb_2024/cleaned_data/{self.data_dir}/context_vector/' + ID + '.pt')
        with open(f'/Users/scottmaran/Code/Python/bdb_2024/cleaned_data/{self.data_dir}/play_info_dict/' + ID, 'rb') as file:
            play_info_dict = pickle.load(file)
        seq_length = play_info_dict['final_frame_in_play']
        player_ids = np.load(f'/Users/scottmaran/Code/Python/bdb_2024/cleaned_data/{self.data_dir}/player_ids/{ID}.npy')

        return X, y, context_vector, seq_length, player_ids, ID
  
  def get_normalize(self):
      all_train_features = torch.zeros((1,self.num_agents*self.feature_size))

      params = {'batch_size': 256,
          'shuffle': False,
          'num_workers': 0}
      dataloader = torch.utils.data.DataLoader(self, **params)

      for batch, _, _, lengths, _, _ in dataloader:
            range_tensor = torch.arange(batch.shape[1])
            padding_msk = (lengths[:, None] > range_tensor).bool()
            all_train_features = torch.concat([all_train_features, batch[padding_msk]], dim=0)
      all_train_features = all_train_features[1:] # remove first zero vector we had

      train_mean = all_train_features.mean(dim=0)
      train_var = all_train_features.var(dim=0)
      min_max_diff = all_train_features.max(dim=0)[0].reshape(1,-1) - all_train_features.min(dim=0)[0].reshape(1,-1)
      normalization_mask = torch.concat([torch.tensor([0,0,0,1,1,1,1,1,1,1,1], dtype=torch.int).repeat((1,self.num_agents))],dim=1).squeeze().bool()

      return train_mean, train_var, normalization_mask, min_max_diff
  
  # return every frame with padding removed
  def get_all_features_and_labels(self):
      all_train_features = torch.zeros((1,self.num_agents*self.feature_size))
      all_train_labels = torch.zeros((1,self.num_agents))
      all_train_ids = torch.zeros(1, self.num_agents)
      all_context_vectors = torch.zeros(1, self.context_length)

      params = {'batch_size': 256,
          'shuffle': False,
          'num_workers': 0}
      dataloader = torch.utils.data.DataLoader(self, **params)

      for x, y, context_vector, lengths, player_ids, _ in dataloader:
            range_tensor = torch.arange(x.shape[1])
            padding_msk = (lengths[:, None] > range_tensor).bool()
            repeated_ids = torch.repeat_interleave(player_ids, lengths, dim=0)  #player_ids = (batch, 23) - want (batch*[each-seq-len], 23)

            all_train_features = torch.concat([all_train_features, x[padding_msk]], dim=0)
            all_train_labels = torch.concat([all_train_labels, y[padding_msk]], dim=0)
            all_train_ids = torch.concat([all_train_ids, repeated_ids], dim=0)      
            all_context_vectors = torch.concat([all_context_vectors, context_vector[padding_msk]], dim=0)

      all_train_features = all_train_features[1:] # remove first zero vector we had
      all_train_labels = all_train_labels[1:] # remove first zero vector we had
      all_train_ids = all_train_ids[1:]
      all_context_vectors = all_context_vectors[1:]

      return all_train_features, all_train_labels, all_context_vectors, all_train_ids
       

'''
There is no padding remaining in the samples in this dataset - covered in preprocessing
'''
class SingleFrameContextDataset(torch.utils.data.Dataset):
  # takes as input json file
  def __init__(self, features : torch.Tensor, labels : torch.Tensor, context: torch.Tensor, player_ids : torch.Tensor):
        self.features = features
        self.labels = labels
        self.context = context
        self.player_ids = player_ids

  def __len__(self):
        'Denotes the total number of samples'
        return self.features.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        return self.features[index], self.labels[index], self.context[index], [], self.player_ids[index], []
  

