import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATConv, global_mean_pool, TopKPooling
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch_geometric.utils import get_laplacian

class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(SimpleGNN, self).__init__()
        self.conv1 = ChebConv(input_dim, hidden_dim1, K=2)
        self.conv2 = GATConv(hidden_dim1, hidden_dim2)
        self.lin1 = torch.nn.Linear(hidden_dim2, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        print(x.shape, edge_index, edge_attr)
        exit()
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)  
        x = F.dropout(x, training=self.training, p=0.35)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = self.lin1(x)
        return x

class EnhancedGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, lstm_hidden_dim):
        super(EnhancedGNN, self).__init__()
        # GNN layers
        self.conv1 = ChebConv(input_dim, hidden_dim1, K=2)
        self.conv2 = GATConv(hidden_dim1, hidden_dim2)
        # LSTM layer for edge sequences
        self.lstm = torch.nn.LSTM(input_size=2, hidden_size=lstm_hidden_dim, batch_first=True)
        # Linear layer
        self.lin1 = torch.nn.Linear(hidden_dim2 + lstm_hidden_dim, 1)

    def forward(self, data, edge_seq, edge_seq_lengths):
        x, edge_index = data.x, data.edge_index

        # GNN part
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        graph_embedding = global_mean_pool(x, data.batch)

        # LSTM part
        packed_edges = pack_padded_sequence(edge_seq, edge_seq_lengths, batch_first=True, enforce_sorted=False)
        _, (lstm_hidden, _) = self.lstm(packed_edges)
        lstm_embedding = lstm_hidden[-1]

        # Concatenate GNN output and LSTM output
        combined_embedding = torch.cat((graph_embedding, lstm_embedding), dim=1)

        # Final linear layer
        out = self.lin1(combined_embedding)

        return out


class SimplestGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimplestGNN, self).__init__()
        self.conv = GCNConv(input_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, 1)
        self.dropout_rate = 0.3

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, training=self.training, p=self.dropout_rate)
        x = self.lin(x)
        return x

class HybridGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(HybridGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GATConv(hidden_dim1, hidden_dim2)
        self.conv3 = GCNConv(hidden_dim2, hidden_dim1 // 2)
        self.lin = torch.nn.Linear(hidden_dim1 // 2, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        x = global_mean_pool(x, data.batch)
        x = self.lin(x)
        
        return x

class ImprovedGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(ImprovedGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GATConv(hidden_dim1, hidden_dim2)
        self.conv3 = GATConv(hidden_dim2, hidden_dim1) 
        self.lin1 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.lin2 = torch.nn.Linear(hidden_dim2, 1)  

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training, p=0.3)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        x = global_mean_pool(x, data.batch)  
        x = self.lin1(x)
        x = F.leaky_relu(x)
        x = self.lin2(x)
        return x

class ClimbingRouteGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(ClimbingRouteGNN, self).__init__()
        self.conv1 = ChebConv(input_dim, hidden_dim1, K=2)
        self.conv2 = GATConv(hidden_dim1, hidden_dim2)
        self.pool = TopKPooling(hidden_dim2, ratio=0.5)
        self.intermediate_lin = torch.nn.Linear(hidden_dim2, hidden_dim1)  
        self.lin = torch.nn.Linear(hidden_dim1, 1)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
        x = global_mean_pool(x, batch)
        x = F.relu(self.intermediate_lin(x))  
        x = self.lin(x)
        return x

