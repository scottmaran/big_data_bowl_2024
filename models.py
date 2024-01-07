
import torch
import torch.nn as nn

class transformer_position(nn.Module):
    def __init__(self, pos_df, feature_embed_size=128, dropout=0.2, num_encoder_layers=4, num_att_heads=32, batch_first=True):
        super(transformer_position, self).__init__()
        self.batch_first = batch_first
        self.feature_input_size = 11
        self.pos_size = 2
        self.context_size = 9

        self.feature_embed_size = feature_embed_size
        self.context_embed_size = 18

        self.pos_df = pos_df

        attention_input_dim = self.feature_embed_size
        self.embed_feature_layer = nn.Linear(self.feature_input_size + self.pos_size, self.feature_embed_size)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=attention_input_dim, dropout=dropout, 
                                                        nhead=num_att_heads, batch_first=self.batch_first,
                                                         dim_feedforward=2048, activation='relu', 
                                                         layer_norm_eps=1e-05)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc1 = nn.Linear(attention_input_dim, 64) #fully connected last layer
        self.fc2 = nn.Linear(64, 1)

        self.relu = nn.LeakyReLU()
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, pos_embeddings, context_vec):
        input = torch.concat([x, pos_embeddings], axis=-1)
        feature_embed = self.relu(self.embed_feature_layer(input))
        out = self.encoder(feature_embed)
        out = self.relu(self.fc1(out))
        logits = self.fc2(out)
        return self.softmax(logits.squeeze(-1))
    

class transformer(nn.Module):
    def __init__(self, feature_embed_size=128, dropout=0.2, num_encoder_layers=4, num_att_heads=32, batch_first=True):
        super(transformer, self).__init__()
        self.batch_first = batch_first
        self.feature_input_size = 11

        self.feature_embed_size = feature_embed_size

        attention_input_dim = self.feature_embed_size# + self.context_embed_size + self.pos_embed_size   # 175

        self.embed_context_layer = nn.Linear(self.context_size, self.context_embed_size)
        self.embed_feature_layer = nn.Linear(self.feature_input_size, self.feature_embed_size)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=attention_input_dim, dropout=dropout, nhead=num_att_heads, batch_first=self.batch_first)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc1 = nn.Linear(attention_input_dim, 64) #fully connected last layer
        self.fc2 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        feature_embed = self.relu(self.embed_feature_layer(x))
        out = self.encoder(feature_embed)
        out = self.relu(self.fc1(out))
        logits = self.fc2(out) # (batch_size,23,1)
        return self.softmax(logits.squeeze(-1))
    