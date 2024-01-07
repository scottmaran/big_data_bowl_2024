import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class trans_single_frame_12_27_pos(nn.Module):
    def __init__(self, pos_df, feature_embed_size=128, dropout=0.2, num_encoder_layers=4, num_att_heads=32, batch_first=True):
        super(trans_single_frame_12_27_pos, self).__init__()
        self.batch_first = batch_first
        self.feature_input_size = 11
        self.pos_size = 2
        self.context_size = 9

        self.feature_embed_size = feature_embed_size
        self.context_embed_size = 18

        self.pos_df = pos_df

        attention_input_dim = self.feature_embed_size
        self.embed_feature_layer = nn.Linear(self.feature_input_size + self.pos_size, self.feature_embed_size)

        #self.embed_layer_2 = nn.Linear(attention_input_dim, attention_input_dim*2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=attention_input_dim, dropout=dropout, 
                                                        nhead=num_att_heads, batch_first=self.batch_first,
                                                         dim_feedforward=2048, activation='relu', 
                                                         layer_norm_eps=1e-05)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc1 = nn.Linear(attention_input_dim, 64) #fully connected last layer
        self.fc2 = nn.Linear(64, 1)

        #self.fc3 = nn.Linear(1+self.context_embed_size+self.feature_input_size, 1)

        self.relu = nn.LeakyReLU()
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, pos_embeddings, context_vec):
        batch_size = x.shape[0]
        input = torch.concat([x, pos_embeddings], axis=-1)  # (batch_size, 23, 13)
        feature_embed = self.relu(self.embed_feature_layer(input))
        out = self.encoder(feature_embed)
        out = self.relu(self.fc1(out))
        logits = self.fc2(out) # (batch_size,23,32)
        return self.softmax(logits.squeeze(-1))
    

class trans_single_frame_12_27(nn.Module):
    def __init__(self, pos_df, feature_embed_size=128, dropout=0.2, num_encoder_layers=4, num_att_heads=32, batch_first=True):
        super(trans_single_frame_12_27, self).__init__()
        self.batch_first = batch_first
        self.feature_input_size = 11
        self.pos_size = 2
        self.context_size = 9

        self.feature_embed_size = feature_embed_size
        self.context_embed_size = 18
        self.pos_embed_size = self.feature_embed_size

        self.pos_df = pos_df

        attention_input_dim = self.feature_embed_size# + self.context_embed_size + self.pos_embed_size   # 175

        self.embed_context_layer = nn.Linear(self.context_size, self.context_embed_size)
        self.embed_feature_layer = nn.Linear(self.feature_input_size, self.feature_embed_size)

        #self.embed_layer_2 = nn.Linear(attention_input_dim, attention_input_dim*2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=attention_input_dim, dropout=dropout, nhead=num_att_heads, batch_first=self.batch_first)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc1 = nn.Linear(attention_input_dim, 64) #fully connected last layer
        self.fc2 = nn.Linear(64, 1)

        self.fc3 = nn.Linear(1+self.context_embed_size+self.feature_input_size, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, pos_embeddings, context_vec):
        batch_size = x.shape[0]

        feature_embed = self.relu(self.embed_feature_layer(x))
        

        #feature_embed = self.relu(self.embed_feature_layer(torch.concat((x, pos_embeddings),dim=-1)))

        #position_embed = self.relu(self.embed_position_layer(pos_embeddings))
        #x = feature_embed + position_embed
        out = self.encoder(feature_embed)
        out = self.relu(self.fc1(out))
        logits = self.fc2(out) # (batch_size,23,1)

        # context_embed = self.relu(self.embed_context_layer(context_vec))    #(batch_size, 18)
        
        # final_out = torch.concat([logits, context_embed.unsqueeze(1).repeat(1,23,1), x], dim=-1) # (batch_size, 23, 1+context_embed+feature_size)
        # final_out = self.fc3(final_out).squeeze(-1)

        return self.softmax(logits.squeeze(-1))
    

class trans_single_frame_13_6_hw(nn.Module):
    def __init__(self, pos_df, feature_embed_size, dropout, num_encoder_layers=4, num_att_heads=32, batch_first=True):
        super(trans_single_frame_13_6_hw, self).__init__()
        self.batch_first = batch_first
        self.feature_input_size = 11

        self.feature_embed_size = feature_embed_size
        self.pos_df = pos_df

        attention_input_dim = self.feature_embed_size# + self.context_embed_size + self.pos_embed_size   # 175

        self.embed_feature_layer = nn.Linear(self.feature_input_size, self.feature_embed_size)

        #self.embed_layer_2 = nn.Linear(attention_input_dim, attention_input_dim*2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=attention_input_dim, dropout=dropout, nhead=num_att_heads, batch_first=self.batch_first)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc1 = nn.Linear(attention_input_dim, 64) #fully connected last layer
        self.fc2 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, pos_embeddings, context_vec):
        feature_embed = self.relu(self.embed_feature_layer(x))
        out = self.encoder(feature_embed)
        out = self.relu(self.fc1(out))
        logits = self.fc2(out) # (batch_size,23,1)
        return self.softmax(logits.squeeze(-1))

class trans_single_frame_13_6_nine(nn.Module):
    def __init__(self, pos_df, feature_embed_size, dropout, num_encoder_layers=4, num_att_heads=32, batch_first=True):
        super(trans_single_frame_13_6_nine, self).__init__()
        self.batch_first = batch_first
        self.feature_input_size = 9
        self.pos_size = 2
        self.context_size = 9

        self.feature_embed_size = feature_embed_size
        self.context_embed_size = 18
        self.pos_embed_size = self.feature_embed_size

        self.pos_df = pos_df

        attention_input_dim = self.feature_embed_size# + self.context_embed_size + self.pos_embed_size   # 175

        self.embed_context_layer = nn.Linear(self.context_size, self.context_embed_size)
        self.embed_feature_layer = nn.Linear(self.feature_input_size, self.feature_embed_size)

        #self.embed_layer_2 = nn.Linear(attention_input_dim, attention_input_dim*2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=attention_input_dim, dropout=dropout, nhead=num_att_heads, batch_first=self.batch_first)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc1 = nn.Linear(attention_input_dim, 64) #fully connected last layer
        self.fc2 = nn.Linear(64, 1)

        self.fc3 = nn.Linear(1+self.context_embed_size+self.feature_input_size, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, pos_embeddings, context_vec):
        feature_embed = self.relu(self.embed_feature_layer(x))
        out = self.encoder(feature_embed)
        out = self.relu(self.fc1(out))
        logits = self.fc2(out) # (batch_size,23,1)
        return self.softmax(logits.squeeze(-1))


''' context embedded with position model '''
class trans_single_frame_12_19_pos_embed_context(nn.Module):
    def __init__(self, feature_input_size, pos_df, num_outputs=1, feature_embed_size=120, dropout=0.2, num_encoder_layers=4, num_att_heads=33, batch_first=True):
        super(trans_single_frame_12_19_pos_embed_context, self).__init__()
        self.batch_first = batch_first
        self.feature_input_size = feature_input_size
        self.feature_embed_size = feature_embed_size
        self.context_embed_size = 23*3
        self.pos_embed_size = 9

        self.pos_df = pos_df

        attention_input_dim = self.feature_embed_size + 3 + self.pos_embed_size   # 175

        self.context_embed = nn.Linear(9, self.context_embed_size)
        self.embed_feature_layer = nn.Linear(feature_input_size, self.feature_embed_size)
        self.embed_layer_2 = nn.Linear(attention_input_dim, attention_input_dim*2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=attention_input_dim*2, dropout=dropout, nhead=num_att_heads, batch_first=self.batch_first)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc1 = nn.Linear(attention_input_dim*2, 64) #fully connected last layer
        self.fc2 = nn.Linear(64, num_outputs)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, pos_embeddings, context_vec):
        x = self.relu(self.embed_feature_layer(x)) # transform each state from (9) to (embed_size, default=128)
        context_embed = self.relu(self.context_embed(context_vec)).reshape(-1,23,3)
        x = torch.concat([x, pos_embeddings, context_embed], dim=-1)
        x = self.relu(self.embed_layer_2(x))
        out = self.encoder(x)#, mask=seq_msk, src_key_padding_mask=padding_mask)  # (batch_size, embed_size)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return self.softmax(out.squeeze(-1))


''' main encoder model used in trans_single_frame_{} models '''
class trans_single_frame_12_12_pos(nn.Module):
    def __init__(self, feature_input_size, num_outputs=1, embed_size=119, dropout=0.1, num_encoder_layers=3, num_att_heads=8, batch_first=True, pos_embeds=True, pos_df=None):
        super(trans_single_frame_12_12_pos, self).__init__()
        self.batch_first = batch_first
        self.feature_input_size = feature_input_size
        self.embed_size = embed_size

        self.pos_embeds = pos_embeds
        self.pos_df = pos_df

        # attention input depends on embedding
        attention_input_dim = self.embed_size
        if self.pos_embeds:
            attention_input_dim = self.embed_size + 9

        self.embed_layer = nn.Linear(feature_input_size, self.embed_size)
        self.embed_layer_2 = nn.Linear(attention_input_dim, attention_input_dim*2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=attention_input_dim*2, dropout=dropout, nhead=num_att_heads, batch_first=self.batch_first)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc1 = nn.Linear(attention_input_dim*2, 64) #fully connected last layer
        self.fc2 = nn.Linear(64, num_outputs)
        #self.fc = nn.Linear(attention_input_dim, num_outputs)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, pos_embeddings):
        x = self.embed_layer(x) # transform each state from (9) to (embed_size, default=128)

        if self.pos_embeds:
            x = torch.concat([x, pos_embeddings], dim=-1)

        x = self.relu(self.embed_layer_2(x))

        out = self.encoder(x)#, mask=seq_msk, src_key_padding_mask=padding_mask)  # (batch_size, embed_size)
        #out = self.fc(out)                                                      # (batch_size, num_ouputs) [default=1]

        out = self.relu(self.fc1(out))
        out = self.fc2(out)

        return self.softmax(out.squeeze(-1))


class NineStateEncoder(nn.Module):
    # input_dim = 9, num_outputs = 1
    def __init__(self, num_outputs, input_dim, dropout=0.1, num_decoder_layers=6, num_att_heads=3, packed=False, batch_first=True, concat=False, device=torch.device('cpu')):
        super(NineStateEncoder, self).__init__()
        self.packed = packed
        self.batch_first = batch_first
        self.concat = concat
        self.device = device

        if self.concat:
            attention_dim = input_dim*2
        else:
            attention_dim = input_dim

        self.pos_encoder = PositionalEncoding(input_dim, concat, device)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=attention_dim, dropout=dropout, nhead=num_att_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_decoder_layers)
        #self.fc1 = nn.Linear(attention_dim, 100) #fully connected last layer
        #self.fc2 = nn.Linear(100, num_outputs)
        self.fc = nn.Linear(attention_dim, num_outputs)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, seq_lengths=None, seq_msk=None, padding_mask=None):
        if self.packed:
            assert seq_lengths != None
            x = pack_padded_sequence(x, seq_lengths, batch_first=self.batch_first, enforce_sorted=False)
        x = self.pos_encoder(x)
        x = x.to(self.device)

        if seq_msk is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            seq_msk = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(self.device)

        out = self.encoder(x, mask=seq_msk, src_key_padding_mask=padding_mask)
        out = self.relu(out)
        return self.softmax(out)

    
''' 
Time encoding:
 - for each agent, encode position in seq. (i.e. time) for kth feature dim

batch_shape = [batch_size, seq_len, 207]

                [batch, seq_len, 23, 9]
'''
class TimeEncoding(nn.Module):
    def __init__(self, d_model:int, concat:bool=False, device:torch.device = torch.device('cpu'), dropout: float = 0.0, max_len: int = 300):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.NUM_FEATURES=9
        self.NUM_PLAYERS=23

        # need to do [1, seq_len, NUM_FEATURES], then concat by NUM_AGENTS
        position = torch.arange(max_len).unsqueeze(1).to(device)   #(max_len, 1)
        div_term = torch.exp(torch.arange(0, self.NUM_FEATURES, 2) * (-math.log(10000.0) / self.NUM_FEATURES)).to(device)  #(d_model/2)
        te = torch.zeros(1, max_len, self.NUM_FEATURES).to(device)
        
        te[0, :, 0::2] = torch.sin(position * div_term) #every even index, (max_len, d_model/2)
        if (self.NUM_FEATURES)%2 != 0:
            te[0, :, 1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            te[0, :, 1::2] = torch.cos(position * div_term) #every odd index
        te = te.repeat(1,1,self.NUM_PLAYERS)

        self.register_buffer('te', te)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        if self.concat:
            # make sure pe is same seq_len size, repeat for same batch size, then concat over feature_dim
            x = torch.cat((x, (self.te[:,:x.size(1),:]).repeat(x.size(0),1,1)), dim=-1)
        else:
            x = x + self.te[:,:x.size(1),:]
        return self.dropout(x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, concat:bool=False, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat

        position = torch.arange(max_len).unsqueeze(1)   #(max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  #(d_model/2)
        #pe = torch.zeros(max_len, 1, d_model)
        pe = torch.zeros(1, max_len, d_model)
        
        pe[0, :, 0::2] = torch.sin(position * div_term) #every even index, (max_len, d_model/2)
        if d_model%2 != 0:
            pe[0, :, 1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            pe[0, :, 1::2] = torch.cos(position * div_term) #every odd index
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, device:torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        if self.concat:
            x = torch.cat((x, (self.pe[:,:x.size(1),:]).repeat(x.size(0),1,1).to(device)), dim=-1)
        else:
            #x = x + (self.pe[:,:x.size(1),:]).to(device)
            x = x + (self.pe[:,:x.size(1),:])
        
        return self.dropout(x)

class transDecoder(nn.Module):
    # decoder_dim = input size into decoder layer
    def __init__(self, num_outputs, decoder_dim, dropout=0.1, num_decoder_layers=6, num_att_heads=9, packed=False, batch_first=True, concat=False, device=torch.device('cpu')):
        super(transDecoder, self).__init__()
        self.packed = packed
        self.batch_first = batch_first
        self.concat = concat
        self.device = device

        if self.concat:
            attention_dim = decoder_dim*2
        else:
            attention_dim = decoder_dim

        self.pos_encoder = TimeEncoding(decoder_dim, concat, device)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=attention_dim, dropout=dropout, nhead=num_att_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        self.fc = nn.Linear(attention_dim, num_outputs) #fully connected last layer
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, seq_lengths=None, target_msk=None, tgt_key_padding_mask=None):
        if self.packed:
            assert seq_lengths != None
            x = pack_padded_sequence(x, seq_lengths, batch_first=self.batch_first, enforce_sorted=False)

        x = self.pos_encoder(x)
        x = x.to(self.device)

        if target_msk is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            target_msk = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(self.device)

        out = self.transformer_decoder(x, memory=torch.zeros(x.shape).to(self.device), tgt_mask=target_msk, tgt_key_padding_mask=tgt_key_padding_mask)
        #out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        
        return self.softmax(out)


class FullStateEncoder(nn.Module):
    def __init__(self, feature_input_size, num_outputs, embed_size=512, dropout=0.1, num_encoder_layers=6, num_att_heads=9, batch_first=True, concat=False):
        super(FullStateEncoder, self).__init__()
        self.batch_first = batch_first
        self.concat = concat
        self.feature_input_size = feature_input_size
        self.embed_size = embed_size

        if self.concat:
            attention_input_dim = embed_size*2
        else:
            attention_input_dim = embed_size

        self.embed_layer = nn.Linear(feature_input_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, concat)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=attention_input_dim, dropout=dropout, nhead=num_att_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc1 = nn.Linear(attention_input_dim, embed_size//2) #fully connected last layer
        self.fc2 = nn.Linear(embed_size//2, num_outputs)
        #self.fc = nn.Linear(attention_input_dim, num_outputs)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    ''' 
    padding_mask is the mask used for each seq in the batch
    seq_msk is the casual mask used within the seq (so not looking into future)
    '''
    def forward(self, x, device=torch.device('cpu'), seq_msk=None, padding_mask=None):
        x = self.embed_layer(x) # transform each state from (207) to (embed_size, default=512)
        x = self.pos_encoder(x, device) # encode positioning of each frame in the sequence

        if seq_msk is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            seq_msk = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(device)

        out = self.encoder(x, mask=seq_msk, src_key_padding_mask=padding_mask)  # (batch_size, max_seq_len, embed_size)
        out = self.relu(self.fc1(out))                                          # (batch_size, max_seq_len, 100)
        out = self.fc2(out)                                                     # (batch_size, max_seq_len, num_outputs)
        return self.softmax(out)
    


''' 
Packing:
How to: https://stackoverflow.com/questions/44131718/padding-time-series-subsequences-for-lstm-rnn-training
        https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
Why computational more efficient for Pytorch: https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch 
'''
class LSTM(nn.Module):
    def __init__(self, num_outputs, input_size, hidden_size, num_layers=1, packed=False, batch_first=True):
        super(LSTM, self).__init__()
        self.num_outputs = num_outputs #number of players we are predicting
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state

        self.packed = packed # bool recording if using packed sequences or not
        self.batch_first = batch_first

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_outputs) #fully connected last layer

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self,x, seq_lengths=None):
        #h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        #c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        if self.packed:
            assert seq_lengths != None
            x = pack_padded_sequence(x, seq_lengths, batch_first=self.batch_first, enforce_sorted=False)
        output, (hn, cn) = self.lstm(x) #lstm with input, hidden, and internal state
        if self.packed:
            output, input_sizes = pad_packed_sequence(output, batch_first=self.batch_first)
        out = self.relu(output)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        
        return self.softmax(out)
    


''' Vision transformer. Too big input size '''
class preTrainedModel(nn.Module):
    def __init__(self):
        super(preTrainedModel, self).__init__()

        # embedding layers to get to (3, 224, 224)
        self.embed_lin1 = nn.Linear(9,1000, bias=False)
        self.embed_lin2 = nn.Linear(23000,224*224, bias=False)

        self.pretrained = vit_l_16(weights=ViT_L_16_Weights)
        #vit_model
        self.mean = torch.tensor([[0.485]], dtype=torch.float16).to(DEVICE)  # for only first dim
        self.std = torch.tensor([[0.229]], dtype=torch.float16).to(DEVICE)

        # change Conv2d layer from (1024, 3, 16, 16) to (1024, 1, 16, 16) and use first channel weights
        trained_weight = self.pretrained.conv_proj.get_parameter('weight')    #(1024,3,16,16)
        self.pretrained.conv_proj = nn.Conv2d(1, 1024, kernel_size=(16, 16), stride=(16, 16), bias=False)
        self.pretrained.conv_proj.get_parameter('weight').data = trained_weight[:,0:1,:,:]
        
        for p in self.pretrained.conv_proj.parameters():    #Conv2d(3, 1024, kernel_size=(16, 16), stride=(16, 16))
            p.requires_grad = False
        for p in self.pretrained.encoder.parameters():    #Encoder
            p.requires_grad = False
        # replace classifier head to 23 outputs, and set grad = True
        self.pretrained.heads = nn.Sequential(nn.Linear(1024, out_features=23, bias=False)) # original is Linear(in_features=1024, out_features=1000, bias=True)
        for p in self.pretrained.heads.parameters():
            p.requires_grad = True

        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=-1)

    # x = [N, 23, 9]
    # mean, std shape = (4)
    def forward(self, x):

        out_embed = self.relu(self.embed_lin1(x))                      # [N, 23, 1000]

        out_embed = out_embed.reshape(-1, 23*1000)          # [N, 23000]

        out_embed = self.embed_lin2(out_embed)              # [N, 65536]
        out_embed = out_embed.reshape(-1, 1, 224, 224)      # [N, 1, 256, 256]

        #out_embed = (out_embed - self.mean.to(x.device))/(self.std.to(x.device))
        out_embed = (out_embed - self.mean)/(self.std)

        vit_out = self.pretrained(out_embed)

        return self.softmax(vit_out)
    

''' 
input passed into forward (batch_size, 23, 9)
- tried 8 epochs, try again longer
'''
# class SingleFrameCNN(nn.Module):
#     # decoder_dim = input size into decoder layer
#     def __init__(self, feature_input_size, num_outputs=23, embed_size=256, dropout=0.0):
#         super(SingleFrameCNN, self).__init__()
#         self.feature_input_size = feature_input_size
#         self.embed_size = embed_size
#         self.hidden_dim = 256

#         self.embed_layer = nn.Linear(self.feature_input_size, self.embed_size)
#         self.conv1 = nn.Conv2d(in_channels=23, out_channels=64, kernel_size=(3,3))
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3))   # [N, 128, 8, 8]
#         self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3))   # [N, 256, 4, 4]
#         self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3))   # [N, 512, 2, 2]
#         self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(2,2))   # [N, 512, 1, 1]
#         self.fc1 = nn.Linear(512, self.hidden_dim)
#         self.fc2 = nn.Linear(self.hidden_dim, num_outputs)

#         self.pool3 = nn.MaxPool2d(3, stride=1)
#         self.relu = nn.LeakyReLU()
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         x = self.embed_layer(x).reshape(-1, 23, 16, 16)     # [batch_size, 23,16,16]
#         out = self.pool3(self.relu(self.conv1(x)))      # [N, 64, 12, 12]
#         out = self.pool3(self.relu(self.conv2(out)))      # [N, 128, 8, 8]
#         out = self.pool3(self.relu(self.conv3(out)))      # [N, 256, 4, 4]
#         out = self.relu(self.conv4(out))                    # [N, 512, 2, 2]
#         out = self.relu(self.conv5(out))                    # [N, 512, 1, 1]

#         out = out.reshape(-1, 512)      # [N, 512]
#         out = self.relu(self.fc1(out))  # [N, hidden_dim]
#         out = self.fc2(out)             # [N, 23]

#         return self.softmax(out)

'''
# one attempt, didn't work well
class SingleFrameCNN(nn.Module):
    # decoder_dim = input size into decoder layer
    def __init__(self, num_outputs, input_dim, device=torch.device('cpu')):
        super(SingleFrameCNN, self).__init__()
        self.device = device
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3,9))   # [N, 252, 21, 1]
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5,1))   # [N, 504, 17, 1]
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(17,1))   # [N, 1016, 1, 1]
        self.fc1 = nn.Linear(512, num_outputs)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.reshape(-1, 512)   # [N, 1016]
        out = self.fc1(out)
        return self.softmax(out)
'''