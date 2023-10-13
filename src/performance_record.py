from model.CancerGATE_tensor import *
from gatev2_utils import *
from preprocessing.Preprocess import load_preprocess_results, load_emogi_result, load_preprocess_results_meth

import pandas as pd
import numpy as np
import pickle
import networkx as nx
from tqdm import tqdm

import dgl
import tensorflow as tf


class PerformanceRecoder:
    def __init__(self, cancer_type, network_name, dim_hiddens, total_epoch):
        self.cancer_type = cancer_type
        self.network_name = network_name
        self.ncg_symbol, self.oncokb_unique, self.negative_symbol = get_symbol_list(cancer_type)
        self.train_index_list, self.train_label_list, self.test_index_list, self.test_label_list, self.gene_list = \
            get_index_list(self.ncg_symbol, self.oncokb_unique, self.negative_symbol, self.cancer_type, self.network_name)
        self.oncokb_unique_index, self.ncg_index, self.total_negative_index, self.gene_list = \
            get_symbol_index(self.gene_list, self.ncg_symbol, self.oncokb_unique)
        self.gpu_usage = False


        # # edge and node information for network construction
        # if 'pancancer' in cancer_type:
        #     if 'tcga_only' in cancer_type:
        #         data_type = 'tcga_only'
        #         self.input_dict = load_emogi_result(data_type)
        #     elif 'tcga_gtex' in cancer_type:
        #         data_type = 'tcga_gtex'
        #         self.input_dict = load_emogi_result(data_type)
        # else:
        self.input_dict = load_preprocess_results_meth(self.cancer_type, self.network_name)
        self.feature_dict = self.input_dict['subtype_x']
        for subtype in self.feature_dict:
            self.feature_dict[subtype] = tf.convert_to_tensor(self.feature_dict[subtype], dtype=tf.float32)
        humannet_edges_node1, humannet_edges_node2 = self.input_dict['edge_index']

        # create network
        self.humannet_dgl = dgl.graph((humannet_edges_node1, humannet_edges_node2))

        # case_name_list = ['Normal', "Her2", 'LumA', 'LumB', "Basal", "Tumor"]
        self.case_name_list = ['Normal', "Tumor"]
        self.subtype = None

        self.dim_hiddens = dim_hiddens  # calculated from performance check
        self.num_layers = len(self.dim_hiddens) - 1

        self.heads = [8, 8]
        self.activation = tf.keras.activations.relu
        self.total_epoch = total_epoch

        self.model_name = 'GATE'

        self.g = self.humannet_dgl
        self.g = dgl.remove_self_loop(self.g)
        self.g = dgl.add_self_loop(self.g)

        self.input_argue_list = ['combine']
        self.set_result()

        self.perturbed_labels = None
        # self.load_perturbed_network()

        self.perturbed_features = None
        # self.load_perturbed_features()


    def set_subtype(self, subtype):
        self.subtype = subtype

    def set_input_argue_list(self, input_argue_list):
        self.input_argue_list = input_argue_list

    def set_result(self):
        self.result = pd.DataFrame(
            columns=['input_argue', 'subtype', 'CV', 'model_type', 'pos_only', 'train_auroc', 'train_auprc',
                     'test_auroc', 'test_auprc'])

    def load_perturbed_network(self):
        self.perturbed_labels = list()
        with open("../data/network/{}_{}_perturbed_networks.pickle".format(self.cancer_type, self.network_name), "rb") as f:
            perturbed_networks = pickle.load(f)
            for i in range(6):
                perturbed_label = nx.to_scipy_sparse_matrix(perturbed_networks[i],
                                                            nodelist=range(len(perturbed_networks[i].nodes)))
                self.perturbed_labels.append(perturbed_label)


    def load_perturbed_features(self):
        self.perturbed_features = dict()
        for case_name in self.case_name_list:
            with open("../data/expression/{}_{}_perturbed_features_{}.pickle".format(self.cancer_type, self.network_name,
                                                                                     case_name.lower()), "rb") as f:
                self.perturbed_features[case_name] = pickle.load(f)

    def main(self):
        print('start')
        # self.get_performance("full")

        print('Start ablation test')
        # self.get_performance("feature_only")
        self.get_performance("structure_only")
        print('Finish ablation test')

        # perturbation test - feature
        # print('Start perturbation test - feature')
        # self.get_performance("feature_25")
        # self.get_performance("feature_50")
        # self.get_performance("feature_75")
        # self.get_performance("feature_100")
        # print('Finish perturbation test - feature')

        # perturbation test - network
        print('Start perturbation test - network')
        # self.get_performance("network_25")
        # self.get_performance("network_50")
        # self.get_performance("network_75")
        # self.get_performance("network_100")
        print('Finish perturbation test - network')

        # perturbation test - both
        print('Start perturbation test - both')
        # self.get_performance("both_25")
        # self.get_performance("both_50")
        # self.get_performance("both_75")
        # self.get_performance("both_100")
        print('Finish perturbation test - both')

        # perturbation test - both
        print('Start perturbation test - random network')
        # self.get_performance("both-random_network_100")
        # self.get_performance("network-random_network_100")
        print('Finish perturbation test - both')

        print('done')

    def get_performance(self, model_type):
        for input_argue in self.input_argue_list:
            for cv in range(10):
                input_dict, model_dict = self.load_models(cv, input_argue, model_type)
                structure_cossim_df, normal_attention_df, tumor_attention_df = self.get_cossim_df(model_dict,
                                                                                                input_dict)

                type_info = [input_argue, self.subtype, cv, self.cancer_type, self.network_name, model_type]
                # self.result.loc[len(self.result)] = type_info + ['selected'] + metric_result
                # self.result.loc[len(self.result)] = type_info + ['pos_only'] + pos_only_metric_result

                self.write_cossim_csv(type_info, structure_cossim_df)
                self.write_attention_csv(type_info, normal_attention_df, 'normal')
                self.write_attention_csv(type_info, tumor_attention_df, 'tumor')

    @staticmethod
    def write_cossim_csv(type_info, structure_cossim_df):
        type_info = [str(info) for info in type_info]
        type_name = '_'.join(type_info)
        structure_cossim_df.to_csv('../result/cossim_csv/{}_meth.tsv'.format(type_name), sep='\t')

    @staticmethod
    def write_attention_csv(type_info, attention_df, subtype):
        type_info = [str(info) for info in type_info]
        type_name = '_'.join(type_info)
        attention_df.to_csv('../result/cossim_csv/{}_{}_attention_meth.tsv'.format(type_name, subtype), sep='\t')


    def load_models(self, cv, input_argue, model_type):
        model_dict = dict()
        input_dict = dict()
        for case_name in self.case_name_list:
            if model_type == 'structure_only':
                g = self.g
                combine_input = self.feature_dict[case_name]
            elif 'both-random' in model_type:
                pert_percent = int(int(model_type.split('_')[-1]) / 25)
                edge_list = self.get_edge_list(self.perturbed_labels[-1])
                g = self.create_input_network(edge_list)
                combine_input = self.perturbed_features[case_name][pert_percent][0]
            elif 'network-random_network' in model_type:
                edge_list = self.get_edge_list(self.perturbed_labels[-1])
                g = self.create_input_network(edge_list)
                combine_input = self.feature_dict[case_name]
            elif 'both_' in model_type:
                pert_percent = int(int(model_type.split('_')[-1]) / 25)
                edge_list = self.get_edge_list(self.perturbed_labels[pert_percent])
                g = self.create_input_network(edge_list)
                combine_input = self.perturbed_features[case_name][pert_percent][0]
            elif 'network' in model_type:
                pert_percent = int(int(model_type.split('_')[-1]) / 25)
                edge_list = self.get_edge_list(self.perturbed_labels[pert_percent])
                g = self.create_input_network(edge_list)
                combine_input = self.feature_dict[case_name]
            elif 'feature' in model_type:
                pert_percent = int(int(model_type.split('_')[-1]) / 25)
                g = self.g
                combine_input = self.perturbed_features[case_name][pert_percent][0]

            in_dims, train_input = check_input_argue(input_argue, combine_input)
            model = CancerGATE(g, self.dim_hiddens, self.heads, self.activation, in_dims)
            condition_list = [cv, self.cancer_type, self.network_name, input_argue,
                              self.model_name, case_name, self.total_epoch,
                              ' '.join([str(dim) for dim in self.dim_hiddens]), model_type]
            condition_list = [str(condition) for condition in condition_list]
            condition_name = '_'.join(condition_list)
            checkpoint_path = "../result/checkpoints/CV{} checkpoints_meth/model.ckpt".format(condition_name)
            model.load_weights(checkpoint_path).expect_partial()

            input_dict[case_name] = train_input
            model_dict[case_name] = model
        return input_dict, model_dict

    @staticmethod
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

    @staticmethod
    def get_edge_list(adj):
        coords, values, shape = sparse_to_tuple(adj)

        # create positive set for train, validation, test from
        coords = coords.tolist()
        positive_set = np.array([coo for coo in coords if coo[0] < coo[1]])
        return positive_set

    def get_cossim_df(self, model_dict, input_dict):
        normal_model = model_dict['Normal']
        normal_data = input_dict['Normal']
        normal_attention, normal_attention_df = get_attention_adj(normal_model, normal_data)

        tumor_model = model_dict[self.subtype]
        tumor_data = input_dict[self.subtype]
        tumor_attention, tumor_attention_df = get_attention_adj(tumor_model, tumor_data)
        cossim_list = get_cossim_list(normal_attention, tumor_attention)

        cos_sim_df = pd.DataFrame(data={self.subtype: cossim_list}, index=self.gene_list)

        return cos_sim_df, normal_attention_df, tumor_attention_df

    def get_auroc(self, cos_sim_df):
        cos_sim_df.index.names = ['symbol']
        cos_sim_df = cos_sim_df.sort_index(axis=0)

        data = 1 - cos_sim_df[self.subtype]
        train_auprc = average_precision_score(self.train_label_list, data.iloc[self.train_index_list])
        train_auroc = roc_auc_score(self.train_label_list, data.iloc[self.train_index_list])
        test_auprc = average_precision_score(self.test_label_list, data.iloc[self.test_index_list])
        test_auroc = roc_auc_score(self.test_label_list, data.iloc[self.test_index_list])

        return [train_auroc, train_auprc, test_auroc, test_auprc]

    def get_auroc_pos_only(self, cos_sim_df):
        cos_sim_df.index.names = ['symbol']
        cos_sim_df = cos_sim_df.sort_index(axis=0)

        pos_only_train_index_list, pos_only_train_label_list, pos_only_test_index_list, pos_only_test_label_list = \
            get_pos_only_dataset(self.oncokb_unique_index, self.ncg_index, self.total_negative_index)

        data = 1 - cos_sim_df[self.subtype]
        train_auprc = average_precision_score(pos_only_train_label_list, data.iloc[pos_only_train_index_list])
        train_auroc = roc_auc_score(pos_only_train_label_list, data.iloc[pos_only_train_index_list])
        test_auprc = average_precision_score(pos_only_test_label_list, data.iloc[pos_only_test_index_list])
        test_auroc = roc_auc_score(pos_only_test_label_list, data.iloc[pos_only_test_index_list])

        return [train_auroc, train_auprc, test_auroc, test_auprc]

    def print_metric_average_std(self, total_metric):
        for key in total_metric.keys():
            print(key)
            cv_metric = total_metric[key]
            train_auprc_list = list()
            train_auroc_list = list()
            test_auprc_list = list()
            test_auroc_list = list()
            for cv in range(5):
                metric_result = cv_metric[cv]
                train_auprc, train_auroc, test_auprc, test_auroc = metric_result['Tumor']
                train_auprc_list.append(train_auprc)
                train_auroc_list.append(train_auroc)
                test_auprc_list.append(test_auprc)
                test_auroc_list.append(test_auroc)

            train_auprc_list = np.array(train_auprc_list)
            train_auroc_list = np.array(train_auroc_list)
            test_auprc_list = np.array(test_auprc_list)
            test_auroc_list = np.array(test_auroc_list)

            print('train auprc\tavg:{}\tstd:{}'.format(train_auprc_list.mean(), train_auprc_list.std()))
            print('train auroc\tavg:{}\tstd:{}'.format(train_auroc_list.mean(), train_auroc_list.std()))
            print('test auprc\tavg:{}\tstd:{}'.format(test_auprc_list.mean(), test_auprc_list.std()))
            print('test auroc\tavg:{}\tstd:{}'.format(test_auroc_list.mean(), test_auroc_list.std()))


if __name__ == "__main__":
    network_name = "CPDB"
    cancer_type = "pancancer"
    model_type_list = ["structure_only"]

    ######
    # model parameters
    ######
    input_mode = 'multi'
    input_argue = 'combine'
    model_name = 'GATE'
    head_list = [8, 8]
    dim_hiddens = [128, 300, 100]
    total_epoch = 1000
    subtype_list = ['Tumor', "Normal"]

    peformance_recorder = PerformanceRecoder(cancer_type, network_name, dim_hiddens, total_epoch)
    peformance_recorder.dim_hiddens = dim_hiddens  # calculated from performance check
    peformance_recorder.num_layers = len(peformance_recorder.dim_hiddens) - 1

    peformance_recorder.heads = head_list
    peformance_recorder.activation = tf.keras.activations.relu
    peformance_recorder.total_epoch = total_epoch

    peformance_recorder.model_name = model_name

    peformance_recorder.set_subtype("Tumor")
    peformance_recorder.get_performance("structure_only")
