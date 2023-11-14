import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATConv, global_mean_pool, TopKPooling
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch_geometric.utils import get_laplacian

def aggregate_edge_features_to_nodes(rnn_out, edge_index, num_nodes):
    agg_node_features = torch.zeros((num_nodes, rnn_out.size(-1)), device=rnn_out.device)
    for i, (source_node, target_node) in enumerate(edge_index.t()):
        # Aggregate features from both the source and target of each edge
        agg_node_features[source_node] += rnn_out[i]
        agg_node_features[target_node] += rnn_out[i]
    edge_count = torch.zeros(num_nodes, device=rnn_out.device)
    for node in edge_index.view(-1):
        edge_count[node] += 1
    edge_count[edge_count == 0] = 1
    agg_node_features = agg_node_features / edge_count.unsqueeze(1)
    return agg_node_features

class SequentialRNNGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, rnn_hidden_dim, dropout_rate=0.5):
        super(SequentialRNNGNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.conv1 = ChebConv(input_dim, hidden_dim1, K=2)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim1)
        self.rnn = torch.nn.GRU(input_size=hidden_dim1 * 2, hidden_size=rnn_hidden_dim, batch_first=True)
        self.conv2 = GATConv(rnn_hidden_dim + hidden_dim1, hidden_dim2)
        self.lin1 = torch.nn.Linear(hidden_dim2, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_index = edge_attr.reshape(2, -1)   
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_rate)
        edge_sequence = edge_attr.view(-1, 2).t().long()
        edge_features_sequence = torch.cat((x[edge_sequence[0]], x[edge_sequence[1]]), dim=1)
        rnn_out, _ = self.rnn(edge_features_sequence)
        agg_node_features = aggregate_edge_features_to_nodes(rnn_out, edge_index, num_nodes=x.size(0))
        x = torch.cat((x, agg_node_features), dim=1)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = self.lin1(x)
        return x

class SequentialLSTMGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, lstm_hidden_dim):
        super(SequentialLSTMGNN, self).__init__()
        self.conv1 = ChebConv(input_dim, hidden_dim1, K=2)
        self.lstm = torch.nn.LSTM(input_size=hidden_dim1 * 2, hidden_size=lstm_hidden_dim, batch_first=True)
        self.conv2 = GATConv(lstm_hidden_dim + hidden_dim1, hidden_dim2)
        self.lin1 = torch.nn.Linear(hidden_dim2, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_index = edge_attr.reshape(2, -1)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.5)
        edge_sequence = edge_attr.view(-1, 2).t().long()
        edge_features_sequence = torch.cat((x[edge_sequence[0]], x[edge_sequence[1]]), dim=1)
        lstm_out, (h_n, c_n) = self.lstm(edge_features_sequence)
        agg_node_features = aggregate_edge_features_to_nodes(h_n.squeeze(0), edge_index, num_nodes=x.size(0))
        x = torch.cat((x, agg_node_features), dim=1)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = self.lin1(x)
        return x
    
class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout_rate=0.5):
        super(SimpleGNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.conv1 = GCNConv(input_dim, hidden_dim1, K=2)
        self.conv2 = GATConv(hidden_dim1, hidden_dim2)
        self.lin1 = torch.nn.Linear(hidden_dim2, 1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim1)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim2)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # edge_index = edge_attr.reshape(2, -1)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)  
        x = F.dropout(x, training=self.training, p=self.dropout_rate)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
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

