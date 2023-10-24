from gatev2_utils import *


network_name = "humannet"
cancer_list = ["BRCA",
               "BLCA",
               "LIHC",
               "KIRC",
               "KIRP",
               "STAD",
               "HNSC",
               "CESC",
               "PRAD",
               "COAD",
               "THCA",
               "LUSC",
               "ESCA",
               "UCEC",
               "LUAD"
               ]
model_type_list = ["structure_only"]

######
# model parameters
######
input_mode = 'multi'
input_argue = 'combine'
model_name = 'GATE'
head_list = [8, 8]
dim_hiddens = [128, 300, 100]
total_epoch = 500
subtype_list = ['Tumor', "Normal"]

for cancer_type in cancer_list:
    print("Now: ", cancer_type)
    gpu_usage = True
    activation = tf.keras.activations.relu

    input_dict = load_preprocess_results(cancer_type, network_name)
    feature_dict = input_dict['subtype_x']
    for subtype in feature_dict:
      feature_dict[subtype] = tf.convert_to_tensor(feature_dict[subtype], dtype=tf.float32)

    # create network
    humannet_edges_node1, humannet_edges_node2 = input_dict['edge_index']
    humannet_dgl = dgl.graph((humannet_edges_node1, humannet_edges_node2))
    if gpu_usage:
      humannet_dgl = humannet_dgl.to('/gpu:0')
    else:
      pass

    # total adj as label
    gene_counts = len(set(humannet_edges_node1) | set(humannet_edges_node2))
    humannet_adj = sp.csr_matrix((np.ones(len(humannet_edges_node1)), (humannet_edges_node1, humannet_edges_node2)), shape=(gene_counts, gene_counts))
    labels = tf.sparse.to_dense(convert_sparse_matrix_to_sparse_tensor(humannet_adj))
    positive_set, negative_set, train_edges, valid_pos, valid_neg, test_pos, test_neg = data_split(humannet_adj, 0.05, 0.1)

    # 2. data load
    subtype_list = list(feature_dict.keys())

    pos_weight = float(humannet_adj.shape[0] * humannet_adj.shape[0] - humannet_adj.sum()) / humannet_adj.sum()
    norm = humannet_adj.shape[0] * humannet_adj.shape[0] / float((humannet_adj.shape[0] * humannet_adj.shape[0] - humannet_adj.sum()) * 2)

    # 3. train model
    from model.CancerGATE_tensor import *
    from optimization import *
    from train import *

    grad_tool = Grads()
    grad_tool.set_labels(labels)
    grad_tool.set_norm(norm)
    grad_tool.set_pos_weight(pos_weight)

    from collections import defaultdict
    perform_dict = defaultdict(list)

    g = create_input_network(train_edges)
    gate_train_result = {'CV':list(), 'subtype':list(), 'train auroc':list(), 'train auprc':list(),
                          'test auroc':list(), 'test auprc':list()}

    for subtype in subtype_list:
        combine_input = feature_dict[subtype]
        in_dims, train_input = check_input_argue(input_argue, combine_input)
        for cv in range(10):
            optimizer = tf.keras.optimizers.Adam(weight_decay=0.005)
            model = CancerGATE(g, dim_hiddens, head_list, activation, in_dims)

            train_structure(model, train_input, optimizer, grad_tool, total_epoch)
            model.save_weights(
                "../result/checkpoints/CV{}_{}_{}_{}_{}_{}_{}_{}_structure_only checkpoints/model.ckpt".format(str(cv),
                                                                                                               cancer_type,
                                                                                                               network_name,
                                                                                                               input_argue,
                                                                                                               model_name,
                                                                                                               subtype,
                                                                                                               total_epoch,
                                                                                                               ' '.join(
                                                                                                                   [str(dim) for
                                                                                                                    dim in
                                                                                                                    dim_hiddens])))
            train_roc_curr, train_ap_curr = get_roc_score(model, train_input, valid_pos, valid_neg)
            test_roc_curr, test_ap_curr = get_roc_score(model, train_input, test_pos, test_neg)
            gate_train_result['CV'].append(cv)
            gate_train_result['subtype'].append(subtype)
            gate_train_result['train auroc'].append(train_roc_curr)
            gate_train_result['train auprc'].append(train_ap_curr)
            gate_train_result['test auroc'].append(test_roc_curr)
            gate_train_result['test auprc'].append(test_ap_curr)
            print("test roc:", test_roc_curr, "\ntest ap:", test_ap_curr)
            cv += 1
    gate_train_result = pd.DataFrame(gate_train_result)
    gate_train_result.to_csv("../result/performance/gate_train_result_{}_{}_{}.tsv".format(cancer_type, network_name, total_epoch), sep='\t', index=False)

# for cancer_type in cancer_list:
    # 3. performance test
    from performance_record import *
    peformance_recorder = PerformanceRecoder(cancer_type, network_name, dim_hiddens, total_epoch)
    peformance_recorder.dim_hiddens = dim_hiddens  # calculated from performance check
    peformance_recorder.num_layers = len(peformance_recorder.dim_hiddens) - 1

    peformance_recorder.heads = head_list
    peformance_recorder.activation = tf.keras.activations.relu
    peformance_recorder.total_epoch = total_epoch

    peformance_recorder.model_name = model_name

    peformance_recorder.set_subtype("Tumor")
    peformance_recorder.get_performance("structure_only")

# for cancer_type in cancer_list:
    from performance_summary import *
    ps = Performance_summary(cancer_type, network_name)
    ncg_symbol, oncokb_unique, negative_symbol = get_symbol_list(ps.cancer_type)
    train_index_list, train_label_list = ps.get_train_answer_index(ncg_symbol, negative_symbol)
    test_index_list, test_label_list = ps.get_test_answer_index(ncg_symbol, oncokb_unique, negative_symbol)
    result = ps.get_performance_df(model_type_list, train_label_list, train_index_list)
    result.to_csv("../result/performance/{}_{}_{}_train_label_performance.tsv".format(cancer_type, network_name, total_epoch), sep="\t") # performance for 75% labels

    test_perform = result[result['input_argue'] == 'combine_test']
    test_perform.to_csv("../result/performance/{}_{}_{}_test_label_performance.tsv".format(cancer_type, network_name, total_epoch), sep="\t")  # performance for 25% labels

    inde_perform = ps.get_performance_df(model_type_list, test_label_list, test_index_list, True)
    inde_perform.to_csv("../result/performance/{}_{}_{}_independent_label_performance.tsv".format(cancer_type, network_name, total_epoch), sep="\t") # performance for OncoKB

    test_perform = result[result['input_argue'] == 'combine_test']
    test_perform = test_perform['auroc'] + test_perform['auprc']
    best_model = list(test_perform).index(max(list(test_perform)))
    print('best model CV:', best_model)

    # 4. get candidates
    import scipy.stats
    file_name = 'combine_Tumor_{}_{}_{}_structure_only.tsv'.format(best_model, cancer_type, network_name)
    cos_sim_df = pd.read_csv('../result/cossim_csv/{}'.format(file_name), sep='\t')
    score_series = 1 - cos_sim_df['Tumor']

    gene_index = pd.read_csv('../data/gene_list/{}_gene_index_{}.tsv'.format(cancer_type, network_name), sep='\t', index_col=0)
    gene_list = gene_index.index.to_list()

    score_series_norm = (score_series - score_series.mean()) / score_series.std()
    p_value_list = scipy.stats.norm.sf(score_series_norm)

    candidate_result = list()
    ncg_symbol, oncokb_unique, negative_symbol = get_symbol_list(cancer_type)
    for i in range(len(p_value_list)):
        if gene_list[i] in ncg_symbol:
            label = 1
        elif gene_list[i] in negative_symbol:
            label = -1
        else:
            label = 0
        candidate_result.append([gene_list[i], score_series[i], p_value_list[i], label])

    candidate_result_df = pd.DataFrame(candidate_result, columns=['gene', 'score', 'p-value', 'label'])
    candidate_result_df.to_csv('../result/prediction_score/{}_{}_{}_candidate.tsv'.format(best_model, cancer_type, network_name), index=False, sep='\t')
