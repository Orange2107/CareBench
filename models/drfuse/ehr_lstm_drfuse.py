import torch
from torch import nn

class EHRLSTMEncoder(nn.Module):
    """LSTM-based EHR encoder for DRFUSE with disentangled representations"""
    
    def __init__(self, input_size, num_classes,
                 hidden_size=256, num_layers_feat=1,
                 num_layers_shared=1, num_layers_distinct=1,
                 dropout=0.3, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers_feat = num_layers_feat
        self.num_layers_shared = num_layers_shared
        self.num_layers_distinct = num_layers_distinct
        self.bidirectional = bidirectional
        
        self.emb = nn.Linear(input_size, hidden_size)
        
        self.lstm_feat = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers_feat,
            dropout=dropout if num_layers_feat > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.lstm_shared = nn.LSTM(
            input_size=lstm_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers_shared,
            dropout=dropout if num_layers_shared > 1 else 0,
            bidirectional=False,
            batch_first=True
        )
        
        self.lstm_distinct = nn.LSTM(
            input_size=lstm_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers_distinct,
            dropout=dropout if num_layers_distinct > 1 else 0,
            bidirectional=False,
            batch_first=True
        )
        
        self.fc_distinct = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, seq_lengths):
        """
        Args:
            x: [batch_size, seq_len, input_size]
            seq_lengths: [batch_size] sequence lengths
        Returns:
            rep_shared: [batch_size, hidden_size] shared representation
            rep_distinct: [batch_size, hidden_size] distinct representation  
            pred_distinct: [batch_size, num_classes] prediction from distinct features
        """
        x = self.emb(x)
        
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_feat, _ = self.lstm_feat(packed_x)
        feat, _ = nn.utils.rnn.pad_packed_sequence(packed_feat, batch_first=True)
        feat = self.dropout(feat)
        
        packed_feat = nn.utils.rnn.pack_padded_sequence(
            feat, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_shared, _ = self.lstm_shared(packed_feat)
        h_shared, _ = nn.utils.rnn.pad_packed_sequence(packed_shared, batch_first=True)
        
        packed_distinct, _ = self.lstm_distinct(packed_feat)
        h_distinct, _ = nn.utils.rnn.pad_packed_sequence(packed_distinct, batch_first=True)
        
        batch_size = x.size(0)
        rep_shared_list = []
        rep_distinct_list = []
        
        for i, length in enumerate(seq_lengths):
            rep_shared_list.append(h_shared[i, length-1, :])
            rep_distinct_list.append(h_distinct[i, length-1, :])
        
        rep_shared = torch.stack(rep_shared_list, dim=0)
        rep_distinct = torch.stack(rep_distinct_list, dim=0)
        
        rep_shared = self.dropout(rep_shared)
        rep_distinct = self.dropout(rep_distinct)
        
        pred_distinct = self.fc_distinct(rep_distinct).sigmoid()
        
        return rep_shared, rep_distinct, pred_distinct 