import numpy as np
import tensorflow as tf
import dgl
import pandas as pd
import random
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def load_preprocess_results(cancer_type, network_name):
    """
    load preprocessing result
    return: 'dict' of edge_index and expression data
    """
    import pickle
    with open("../data/specific_cancer_inputs/{}_input_data_{}.pkl".format(cancer_type, network_name), 'rb') as f:
        return pickle.load(f)



def get_reverse_edge(edge_list):
    return np.array([[edge[1], edge[0]] for edge in edge_list])


def create_input_network(edge_list, gpu_usage=True):
    humannet_dgl = dgl.graph((edge_list[:, 0], edge_list[:, 1]))
    if gpu_usage:
        humannet_dgl = humannet_dgl.to('/gpu:0')
    else:
        pass
    g = humannet_dgl

    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g


def check_input_argue(input_argue, combine_input):
    if input_argue == 'combine':
        in_dims = [combine_input.shape[1]]
        train_input = [combine_input]
    elif input_argue == 'expression':
        input_data = combine_input[:, :-12]
        in_dims = [input_data.shape[1]]
        train_input = [input_data]
    elif input_argue == 'mutation':
        input_data = combine_input[:, -12:]
        in_dims = [input_data.shape[1]]
        train_input = [input_data]
    elif input_argue == 'whole':
        exp_data = combine_input[:, :-12]
        mut_data = combine_input[:, -12:]
        in_dims = [exp_data.shape[1], mut_data.shape[1]]
        train_input = [exp_data, mut_data]
    else:
        in_dims = None
        train_input = None
    return in_dims, train_input


def convert_sparse_matrix_to_sparse_tensor(inputs):
    coo = inputs.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    output = tf.SparseTensor(indices, coo.data.astype('float64'), coo.shape)
    output = tf.dtypes.cast(output, tf.float32)
    return output


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def data_split(adj, validation_rate,
               test_rate):
    coords, values, shape = sparse_to_tuple(adj)

    # create positive set for train, validation, test from
    coords = coords.tolist()
    positive_set = np.array([coo for coo in coords if not coo[0] == coo[1]])# if coo[0] < coo[1]])
    positive_idx = np.array([coo[0] * shape[0] + coo[1] for coo in positive_set])

    np.random.shuffle(positive_set)

    test_num = int(len(positive_set) * test_rate)
    validation_num = int(len(positive_set) * validation_rate)
    test_pos = positive_set[:test_num]
    valid_pos = positive_set[test_num:(test_num + validation_num)]
    train_edges = positive_set[(test_num + validation_num):]

    # create negative set for validation, test
    negative_idx_list = list()

    while len(negative_idx_list) < len(positive_idx):
        i = random.randrange(shape[0])
        j = random.randrange(shape[0])
        if i < j:
            idx = i * shape[0] + j
            if idx not in positive_idx:
                negative_idx_list.append(idx)

    negative_idx = np.array(negative_idx_list)
    negative_set = np.array([[idx // shape[0], idx % shape[0]] for idx in negative_idx])
    test_neg = negative_set[:test_num]
    valid_neg = negative_set[test_num:(test_num + validation_num)]

    return positive_set, negative_set, train_edges, valid_pos, valid_neg, test_pos, test_neg


def accuracy_ae(logits, labels):
    # acc function
    correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(logits), 0.5), tf.float32),
                                  tf.cast(labels, tf.float32))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return acc.numpy().item()


def get_pos_only_dataset(oncokb_unique_index, ncg_index, total_negative_index):
    random.shuffle(total_negative_index)
    train_ratio = len(ncg_index) / (len(ncg_index) + len(oncokb_unique_index))
    train_count = int(train_ratio * len(total_negative_index))
    pos_only_train_negative_index = total_negative_index[:train_count]
    pos_only_test_negative_index = total_negative_index[train_count:]

    pos_only_train_index_list = ncg_index + pos_only_train_negative_index
    pos_only_train_label_list = np.concatenate((np.ones(len(ncg_index)), np.zeros(len(pos_only_train_negative_index))))
    pos_only_test_index_list = oncokb_unique_index + pos_only_test_negative_index
    pos_only_test_label_list = np.concatenate(
        (np.ones(len(oncokb_unique_index)), np.zeros(len(pos_only_test_negative_index))))

    return pos_only_train_index_list, pos_only_train_label_list, pos_only_test_index_list, pos_only_test_label_list


def get_random_dataset(gene_list, negative_symbol, total_negative_index, ncg_index, oncokb_unique_index):
    """
    1. get total negative set (exclude all positive index)
    2. create random negative set as same size as true negative set
    3. create train_index_list, train_label_list, test_index_list, test_label_list as original method
    """
    golden_negative_index = [i for i in range(len(gene_list)) if gene_list[i] in negative_symbol]
    random_negative_index = np.random.choice(total_negative_index, len(golden_negative_index), replace=False)
    random_train_index_list = ncg_index + random_negative_index
    random_train_label_list = np.concatenate((np.ones(len(ncg_index)), np.zeros(len(random_negative_index))))

    random_test_negative_index = set(total_negative_index) - set(random_negative_index)
    random_test_index_list = oncokb_unique_index + random_test_negative_index
    random_test_label_list = np.concatenate(
        (np.ones(len(oncokb_unique_index)), np.zeros(len(random_test_negative_index))))
    return random_train_index_list, random_train_label_list, random_test_index_list, random_test_label_list


def accuracy_cls(logits, labels):
    indices = tf.math.argmax(logits, axis=1)
    label_indices = tf.math.argmax(labels, axis=1)
    acc = tf.reduce_mean(tf.cast(indices == label_indices, dtype=tf.float32))
    return acc.numpy().item()


def evaluate_ae(model, features, labels):
    logits = model(features, training=False)
    return accuracy_ae(logits, labels)


def evaluate_cls(model, features, labels, mask):
    logits = model(features, training=False)
    logits = logits[mask]
    labels = labels[mask]
    return accuracy_cls(logits, labels)


def get_roc_score(model, train_input, edges_pos, edges_neg):
    logits_pos = tf.sigmoid(model.get_reconstructed_edge(train_input, edges_pos)).numpy()
    logits_neg = tf.sigmoid(model.get_reconstructed_edge(train_input, edges_neg)).numpy()

    preds_all = np.hstack([logits_pos, logits_neg])
    labels_all = np.hstack([np.ones(len(logits_pos)), np.zeros(len(logits_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def get_roc_score2(model, g, feature, edges_pos, edges_neg):
    logits_pos = tf.sigmoid(model.get_reconstructed_edge(g, feature, edges_pos)).numpy()
    logits_neg = tf.sigmoid(model.get_reconstructed_edge(g, feature, edges_neg)).numpy()

    preds_all = np.hstack([logits_pos, logits_neg])
    labels_all = np.hstack([np.ones(len(logits_pos)), np.zeros(len(logits_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def get_symbol_list(cancer_type):
    # target list - oncoKB
    file_name = '../data/gene_list/oncoKB_specific_gene_list/{}'
    if 'pancancer' in cancer_type:
        file_name = file_name.format('oncokb_biomarker_drug_associations.tsv')
    else:
        single_file = '{}_oncokb_biomarker_drug_associations.tsv'.format(cancer_type)
        file_name = file_name.format(single_file)
    oncokb_df = pd.read_csv(file_name, sep='\t', index_col=False)
    oncokb_list = oncokb_df['Gene'].to_list()


    # target list - NCGv6
    if 'pancancer' in cancer_type:
        ncg_df = pd.read_csv("../data/gene_list/796true.txt", sep='\t', header=None)
        ncg_symbol = ncg_df[0].to_list()
        ncg_symbol = list(set(ncg_symbol))
        ncg_symbol.sort()
    else:
        ncg_df = pd.read_csv('../data/gene_list/cancer_specific_gene_list/{}_pos.tsv'.format(cancer_type), sep="\t")
        ncg_symbol = ncg_df['symbol'].to_list()
        ncg_symbol = list(set(ncg_symbol))
        ncg_symbol.sort()
    negative_df = pd.read_csv('../data/gene_list/2187_neg.tsv', sep="\t",
                              index_col=False)
    negative_symbol = list(set(negative_df['symbol']))
    random.shuffle(negative_symbol)

    oncokb_unique = set(oncokb_list) - set(ncg_symbol)
    oncokb_unique = list(oncokb_unique)
    return ncg_symbol, oncokb_unique, negative_symbol


def get_specific_symbol_list(cancer_type):
    # target list - oncoKB + cosmic + literature
    file_name = ''

    oncokb_df = pd.read_csv("../data/gene_list/oncoKB_inde.tsv", sep='\t', index_col=False)
    oncokb_list = oncokb_df['symbol'].to_list()

    # target list - NCGv6 cancer specific
    ncg_symbol = pd.read_csv("../data/gene_list/cancer_specific_gene_list/{}_pos.tsv".format(cancer_type), sep='\t')
    ncg_symbol = ncg_symbol['symbol'].to_list()

    # target list - negative set
    negative_df = pd.read_csv("../data/gene_list/2187false.txt", sep='\t', header=None)
    negative_symbol = list(set(negative_df[0]))
    random.shuffle(negative_symbol)

    oncokb_unique = set(oncokb_list) - set(ncg_symbol)
    oncokb_unique = list(oncokb_unique)
    return ncg_symbol, oncokb_unique, negative_symbol

def get_gene_list(cancer_type, network_name):
    gene_index = pd.read_csv('../data/gene_list/{}_gene_index_{}.tsv'.format(cancer_type, network_name), sep='\t',
                             index_col=0)
    gene_list = gene_index.index.to_list()
    return gene_list

def get_index_list(ncg_symbol, oncokb_unique, negative_symbol, cancer_type, network_name):
    gene_index = pd.read_csv('../data/gene_list/{}_gene_index_{}.tsv'.format(cancer_type, network_name), sep='\t', index_col=0)
    gene_list = gene_index.index.to_list()

    train_index_list = list()
    train_label_list = list()
    for i in range(len(gene_list)):
        symbol = gene_list[i]
        if symbol in ncg_symbol:
            train_index_list.append(i)
            train_label_list.append(1)
        elif symbol in negative_symbol:
            train_index_list.append(i)
            train_label_list.append(0)

    test_index_list = list()
    test_label_list = list()
    for i in range(len(gene_list)):
        symbol = gene_list[i]
        if symbol in oncokb_unique:
            test_index_list.append(i)
            test_label_list.append(1)
        elif symbol not in ncg_symbol and symbol not in negative_symbol:
            test_index_list.append(i)
            test_label_list.append(0)

    return train_index_list, train_label_list, test_index_list, test_label_list, gene_list


def get_symbol_index(gene_list, ncg_symbol, oncokb_unique):
    total_negative_index = [i for i in range(len(gene_list)) if
                            gene_list[i] not in ncg_symbol and gene_list[i] not in oncokb_unique]
    ncg_index = [i for i in range(len(gene_list)) if gene_list[i] in ncg_symbol]
    oncokb_unique_index = [i for i in range(len(gene_list)) if gene_list[i] in oncokb_unique]
    return oncokb_unique_index, ncg_index, total_negative_index, gene_list


def get_attention_adj(model, input_data):
    import networkx as nx
    graph = model.g
    total_attention = model.get_attention(input_data)
    src_nodes = graph.edges()[0]
    dst_nodes = graph.edges()[1]

    attention_df = pd.DataFrame(
        {'src': src_nodes.numpy(), 'dst': dst_nodes.numpy(), 'attention': total_attention.numpy()})

    g = nx.from_pandas_edgelist(attention_df, 'src', 'dst', ['attention'], create_using=nx.DiGraph)
    adj = nx.to_pandas_adjacency(g, weight='attention')
    adj = adj.sort_index()
    adj = adj.sort_index(axis=1)

    return adj, attention_df


def get_attention_df(model, input_data):
    import networkx as nx
    graph = model.g
    total_attention = model.get_attention(input_data)
    src_nodes = graph.edges()[0]
    dst_nodes = graph.edges()[1]

    attention_df = pd.DataFrame(
        {'src': src_nodes.numpy(), 'dst': dst_nodes.numpy(), 'attention': total_attention.numpy()})

    return attention_df


def get_attention_network(model, input_data):
    import networkx as nx
    graph = model.g
    total_attention = model.get_attention(input_data)
    src_nodes = graph.edges()[0]
    dst_nodes = graph.edges()[1]

    attention_df = pd.DataFrame(
        {'src': src_nodes.numpy(), 'dst': dst_nodes.numpy(), 'attention': total_attention.numpy()})

    g = nx.from_pandas_edgelist(attention_df, 'src', 'dst', ['attention'], create_using=nx.DiGraph)

    return g


def get_embedding_vector(model, input_data):
    embedding_vector = model.embedding(input_data)
    embedding_vector = tf.reduce_mean(embedding_vector, axis=1)
    return embedding_vector.numpy()


def get_cossim_list(normal_attention, tumor_attention):
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm import tqdm
    tumor_cossim = list()
    for i in tqdm(range(len(normal_attention))):
        cossim = cosine_similarity([normal_attention[i]],
                                   [tumor_attention[i]])
        tumor_cossim.append(cossim[0][0])
    return tumor_cossim


def get_emogi_fp_label_index(file_name):
    import pickle
    with open("../data/gene_list/{}.pkl".format(file_name), "rb") as f:
        result = pickle.load(f)
        return result