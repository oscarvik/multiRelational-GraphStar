import sys
import ssl
from os import path, mkdir
import numpy as np
import pandas as pd
import torch
import trainer
import utils.gsn_argparse as gap
import utils.label_encode_dataset as led
import utils.create_node_embedding as cne
import utils.create_relation_embedding as cre
import utils.misc as misc
from gensim.models import KeyedVectors
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import structured_negative_sampling

ssl._create_default_https_context = ssl._create_unverified_context


def train_val_test_split(dataset, train, val, test):
    # edge_indexes
    dataset.train_pos_edge_index = train.edge_index
    dataset.val_pos_edge_index = val.edge_index
    dataset.test_pos_edge_index = test.edge_index

    # relations
    dataset.train_edge_type = train.edge_type
    dataset.val_edge_type = val.edge_type
    dataset.test_edge_type = test.edge_type

    # negatives
    # nb! add symmetry before
    anti_symmetric_train = torch.flip(dataset.train_pos_edge_index,[0])
    symmetric_train = torch.cat([dataset.train_pos_edge_index, anti_symmetric_train], dim=1)

    neg_train = structured_negative_sampling(symmetric_train)
    dataset.train_neg_edge_index = torch.tensor(
        [list(neg_train[0]), list(neg_train[2])], dtype=torch.long
    ).narrow(1,0,anti_symmetric_train.size(1))
    
    anti_symmetric_val = torch.flip(dataset.val_pos_edge_index,[0])
    symmetric_val = torch.cat([dataset.val_pos_edge_index, anti_symmetric_val], dim=1)

    neg_val = structured_negative_sampling(symmetric_val)
    dataset.val_neg_edge_index = torch.tensor(
        [list(neg_val[0]), list(neg_val[2])], dtype=torch.long
    ).narrow(1,0,anti_symmetric_val.size(1))

    anti_symmetric_test = torch.flip(dataset.test_pos_edge_index,[0])
    symmetric_test = torch.cat([dataset.test_pos_edge_index, anti_symmetric_test], dim=1)
    neg_test = structured_negative_sampling(symmetric_test)
    dataset.test_neg_edge_index = torch.tensor(
        [list(neg_test[0]), list(neg_test[2])], dtype=torch.long
    ).narrow(1,0,anti_symmetric_test.size(1))

    return dataset


def load_data(news_dataset, kg_dataset, dataset_name, hidden=64, node_embedding_size=16, embedding_path="fn_embeddings"):
    print("Loading open KG: " + kg_dataset + "...")
    columns = {
        "FB15k": ["head", "tail", "relation"],
        "FB15k_237": ["head", "relation", "tail"],
    }

    train = pd.read_csv(
        "./data/" + kg_dataset + "/train.txt",
        sep="\t",
        header=None,
        names=columns[kg_dataset],
        engine="python",
    )
    valid = pd.read_csv(
        "./data/" + kg_dataset + "/valid.txt",
        sep="\t",
        header=None,
        names=columns[kg_dataset],
        engine="python",
    )
    test = pd.read_csv(
        "./data/" + kg_dataset + "/test.txt",
        sep="\t",
        header=None,
        names=columns[kg_dataset],
        engine="python",
    )

    print("Loading fake news dataset: " + news_dataset + "...")
    fn_test = pd.read_csv(
        "./data/" + news_dataset + "/train_both.csv",
        sep=",",
        header=0,
        engine="python",
        index_col=0,
    )

    # create model folder
    save_folder = "model_" + misc.get_now()
    print("creating model folder: " + save_folder + "...")
    mkdir(save_folder)

    # needed for embedding all nodes across datasets (will use edge_index to split datasets)
    all_data = pd.concat([train, valid])
    all_data = pd.concat([all_data, test])
    all_data.drop_duplicates(inplace=True)

    entity_id = pd.read_csv('data/FB15k/entities.txt', sep='\t', header=None, names=['entity', 'id'], engine='python')
    entities = entity_id['entity'].values

    news_dataset_entities = np.concatenate([fn_test["fb_head"], fn_test["fb_tail"]])
    news_dataset_entities = np.unique(news_dataset_entities)
    new_entities = np.array([ent for ent in news_dataset_entities if ent not in entities])
    entities = np.concatenate([entities, new_entities])
    entity_ids = torch.arange(0, len(entities), 1, dtype=torch.float)
    
    relation_id = pd.read_csv('data/FB15k/relations.txt', sep='\t', header=None, names=['relation', 'id'], engine='python')
    relations = relation_id['relation'].values


    # fit entity and relation encoder
    le_entity = LabelEncoder()
    le_entity.fit(entities)
    le_relation = LabelEncoder()
    le_relation.fit(relations)


    np.save(path.join(save_folder, "le_relation_classes_" + dataset_name + ".npy"), le_relation.classes_)
    np.save(path.join(save_folder, "le_entity_classes_" + dataset_name + ".npy"), le_entity.classes_)

    train = led.label_encode_dataset(le_entity, le_relation, train, entity_ids)
    valid = led.label_encode_dataset(le_entity, le_relation, valid, entity_ids)
    test = led.label_encode_dataset(le_entity, le_relation, test, entity_ids)

    all_data = led.label_encode_dataset(le_entity, le_relation, all_data, entity_ids)
    news_dataset_x = torch.tensor(le_entity.transform(new_entities), dtype=torch.float)
    all_data.x = torch.cat([all_data.x, news_dataset_x])
    print('all_data.x.size: ', all_data.x.size())
    # create node embeddings if none exists
    cne.create_node_embedding(
        all_data, dataset_name, dimensions=node_embedding_size, workers=4, path=save_folder
    )
    cre.create_relation_embedding(relations, le_relation, dataset_name, dimensions=hidden, path=save_folder)
    embedded_nodes = KeyedVectors.load_word2vec_format(
        "{}/node_embedding_{}_{}.kv".format(save_folder, dataset_name, str(node_embedding_size))
    )
    
    embedded_relations = KeyedVectors.load_word2vec_format(
        "{}/relation_embedding_le_{}_{}.bin".format(save_folder, dataset_name, str(hidden)),
        binary=True,
    )
    # need to sort to get correct indexing
    sorted_embedding = []
    for i in range(0, len(embedded_nodes.vectors)):
        sorted_embedding.append(embedded_nodes.get_vector(str(i)))  # error her
    all_data.x = torch.tensor(sorted_embedding, dtype=torch.float)

    sorted_embedding = []
    for i in range(0, len(embedded_relations.vectors)):
        sorted_embedding.append(embedded_relations.get_vector(str(i)))
    embedded_relations = torch.tensor(sorted_embedding, dtype=torch.float)

    all_data.batch = torch.zeros((1, all_data.num_nodes), dtype=torch.int64).view(-1)

    num_features = all_data.x.shape[-1]
    num_relations = len(np.unique(relations))

    data = train_val_test_split(all_data, train, valid, test)

    data.edge_index = torch.cat(
        [data.train_pos_edge_index, data.val_pos_edge_index, data.test_pos_edge_index],
        dim=1,
    )
    data.edge_type = torch.cat(
        [train.edge_type, valid.edge_type, test.edge_type], dim=0
    )

    return data, num_features, num_relations, embedded_relations, save_folder


def main(_args):
    print(
        "\033[1;32m"
        + "@@@@@@@@@@@@@@@@ Fake News Detection through Multi-Relational Graph Star @@@@@@@@@@@@@@@@"
        + "\033[0m"
    )
    args = gap.parser.parse_args(_args)
    dataset_name = args.dataset + '_' + args.news_dataset
    data, num_features, num_relations, embedded_relations, save_folder = load_data(
        hidden=args.hidden, kg_dataset=args.dataset, news_dataset=args.news_dataset, dataset_name=dataset_name
    )

    embedded_relations.to(args.device)    
    trainer.trainer(
        args,
        dataset_name,
        data,
        num_features=num_features,
        num_relations=num_relations,
        relation_embeddings=embedded_relations,
        num_epoch=args.epochs,
        save_folder=save_folder
    )
    

if __name__ == "__main__":
    main(sys.argv[1:])
