import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATConv, TransformerConv, GlobalAttention, global_mean_pool, global_max_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch_geometric.utils import get_laplacian
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d

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
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, conv_types, k_values, dropout_rate1, dropout_rate2):
        super(SimpleGNN, self).__init__()
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.conv1 = conv_types[0](input_dim, hidden_dim1, K=k_values[0])
        self.conv2 = conv_types[1](hidden_dim1, hidden_dim2, K=k_values[1])
        self.conv3 = conv_types[2](hidden_dim2, hidden_dim3, K=k_values[2])
        self.lin1 = torch.nn.Linear(hidden_dim3, 1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim1)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim2)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim3)
        self.shortcut = torch.nn.Linear(input_dim, hidden_dim3)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_index = edge_attr.reshape(2, -1) # same as edge_index but ordered, should not be important as the order should not matter
        identity = x 
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)  
        x = F.dropout(x, training=self.training, p=self.dropout_rate1)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)  
        x = F.dropout(x, training=self.training, p=self.dropout_rate2)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        identity = self.shortcut(identity)
        x += identity  
        x = global_max_pool(x, data.batch)  
        x = self.lin1(x)
        return x

