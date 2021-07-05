from torch_geometric.data import Data
import torch


def label_encode_dataset(le_entity, le_relation, data, entity_ids):
    head = data["head"].values
    tail = data["tail"].values
    relations = data["relation"].values

    # string list to int array using LabelEncoder on complete data set
    heads = le_entity.transform(head)
    tails = le_entity.transform(tail)
    relations = le_relation.transform(relations)

    edge_attributes = torch.tensor(relations, dtype=torch.long)
    edge_index = torch.tensor([heads, tails], dtype=torch.long)

    return Data(x=entity_ids, edge_type=edge_attributes, edge_index=edge_index)
