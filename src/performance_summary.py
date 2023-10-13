import os
from gatev2_utils import *
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

class Performance_summary():
    def __init__(self, cancer_type, network_name):
        self.input_argue = 'combine'
        self.subtype = 'Tumor'
        self.cancer_type = cancer_type
        self.network_name = network_name


    def get_test_answer_index(self, ncg_symbol, oncokb_unique, negative_symbol):
        """
            Get index_list and label_list from ncg, oncokb genes and negative genes
            :return: index_list, label_list
            """

        gene_index = pd.read_csv('../data/gene_list/{}_gene_index_{}.tsv'.format(self.cancer_type, self.network_name), sep='\t', index_col=0)
        gene_list = gene_index.index.to_list()

        index_list = list()
        label_list = list()
        for i in range(len(gene_list)):
            symbol = gene_list[i]
            if symbol in oncokb_unique:
                index_list.append(i)
                label_list.append(1)
            elif symbol not in negative_symbol and symbol not in ncg_symbol:
                index_list.append(i)
                label_list.append(0)
        return index_list, label_list

    def get_train_answer_index(self, ncg_symbol, negative_symbol):
        """
            Get index_list and label_list from ncg, oncokb genes and negative genes
            :return: index_list, label_list
            """

        gene_index = pd.read_csv('../data/gene_list/{}_gene_index_{}.tsv'.format(self.cancer_type, self.network_name), sep='\t', index_col=0)
        gene_list = gene_index.index.to_list()

        index_list = list()
        label_list = list()
        for i in range(len(gene_list)):
            symbol = gene_list[i]
            if symbol in ncg_symbol:
                index_list.append(i)
                label_list.append(1)
            elif symbol in negative_symbol:
                index_list.append(i)
                label_list.append(0)
        return index_list, label_list


    def get_random_dataset(self, gene_list, negative_symbol, total_negative_index, ncg_index, oncokb_unique_index):
        """
        1. get total negative set (exclude all positive index)
        2. create random negative set as same size as true negative set
        3. create train_index_list, train_label_list, test_index_list, test_label_list as original method
        """
        golden_negative_index = [i for i in range(len(gene_list)) if gene_list[i] in negative_symbol]
        random_negative_index = np.random.choice(total_negative_index, len(golden_negative_index))
        random_train_index_list = ncg_index + list(random_negative_index)
        random_train_label_list = np.concatenate((np.ones(len(ncg_index)), np.zeros(len(random_negative_index))))

        random_test_negative_index = set(total_negative_index) - set(random_negative_index)
        random_test_index_list = oncokb_unique_index + list(random_test_negative_index)
        random_test_label_list = np.concatenate(
            (np.ones(len(oncokb_unique_index)), np.zeros(len(random_test_negative_index))))
        return random_train_index_list, random_train_label_list, random_test_index_list, random_test_label_list


    def get_total_answer_index(self, ncg_symbol, oncokb_unique, negative_symbol):
        """
        Get index_list and label_list from ncg, oncokb genes and negative genes
        :return: index_list, label_list
        """

        gene_index = pd.read_csv('../data/gene_list/{}_gene_index_{}.tsv'.format(self.cancer_type, self.network_name), sep='\t', index_col=0)
        gene_list = gene_index.index.to_list()

        index_list = list()
        label_list = list()
        for i in range(len(gene_list)):
            symbol = gene_list[i]
            if symbol in ncg_symbol or symbol in oncokb_unique:
                index_list.append(i)
                label_list.append(1)
            elif symbol in negative_symbol:
                index_list.append(i)
                label_list.append(0)
        return index_list, label_list

    def get_total_positive_answer_index(self, ncg_symbol, oncokb_unique):
        """
        Get index_list and label_list from ncg, oncokb genes and negative genes
        :return: index_list, label_list
        """

        gene_index = pd.read_csv('../data/gene_list/{}_gene_index_{}.tsv'.format(self.cancer_type, self.network_name), sep='\t', index_col=0)
        gene_list = gene_index.index.to_list()

        index_list = list()
        label_list = list()
        for i in range(len(gene_list)):
            symbol = gene_list[i]
            if symbol in ncg_symbol or symbol in oncokb_unique:
                index_list.append(i)
                label_list.append(1)
            else:
                index_list.append(i)
                label_list.append(0)
        return index_list, label_list


    def get_symbol_index(self, ncg_symbol, oncokb_unique):
        gene_index = pd.read_csv('../data/gene_list/{}_gene_index_{}.tsv'.format(self.cancer_type, self.network_name), sep='\t', index_col=0)
        gene_list = gene_index.index.to_list()

        total_negative_index = [i for i in range(len(gene_list)) if
                                gene_list[i] not in ncg_symbol and gene_list[i] not in oncokb_unique]
        ncg_index = [i for i in range(len(gene_list)) if gene_list[i] in ncg_symbol]
        oncokb_unique_index = [i for i in range(len(gene_list)) if gene_list[i] in oncokb_unique]
        return oncokb_unique_index, ncg_index, total_negative_index, gene_list


    def get_performance_df(self, model_type_list, label_list, index_list, independent_test=False):
        result = pd.DataFrame(columns=['input_argue', 'model_type', 'auroc', 'auprc'])
        for model_type in model_type_list:
            if independent_test:
                total_index_df = self.get_performance_independent(model_type, label_list, index_list)
            else:
                total_index_df = self.get_performance(model_type, label_list, index_list)
            result = pd.concat([result, total_index_df], ignore_index=True)
        return result


    def get_performance(self, model_type, label_list, index_list):
        total_index_df = pd.DataFrame(columns=['input_argue', 'model_type', 'auroc', 'auprc'])

        input_argue = 'combine'

        cossim_list = os.listdir('../result/cossim_csv')
        cossim_list.sort()
        for cv in range(10):
            type_info = [input_argue, self.subtype, str(cv), self.cancer_type, self.network_name, model_type]
            file_name = '_'.join(type_info) + '_meth.tsv'
        # for file_name in cossim_list:
        #     if file_name.endswith('{}.tsv'.format(model_type)):
        #         if model_type == 'network_100' and "random" in file_name:
        #             continue
            cos_sim_df = pd.read_csv('../result/cossim_csv/{}'.format(file_name), sep='\t')
            score_series = 1 - cos_sim_df['Tumor']
            # input_argue = file_name.split("_")[0]

            # total label and score to performance
            auroc = roc_auc_score(label_list, score_series.iloc[index_list])
            auprc = average_precision_score(label_list, score_series.iloc[index_list])
            total_index_df.loc[len(total_index_df)] = [input_argue, model_type, auroc, auprc]

            # split label and scores to train set and test set as other training models
            train_input_feature, test_input_feature, train_input_label_list, test_input_label_list = \
                train_test_split(score_series.iloc[index_list], label_list, test_size=0.25, stratify=label_list)
            train_input_label_list = np.array(train_input_label_list)
            test_input_label_list = np.array(test_input_label_list)

            # auroc auproc for test labels
            auroc = roc_auc_score(test_input_label_list, test_input_feature)
            auprc = average_precision_score(test_input_label_list, test_input_feature)
            total_index_df.loc[len(total_index_df)] = [input_argue + '_test', model_type, auroc, auprc]

            # auroc auprc for validation labels
            kf = StratifiedKFold(n_splits=5, shuffle=True)
            for train_index, test_index in kf.split(train_input_feature, train_input_label_list):
                X_train, X_test = train_input_feature.iloc[train_index], train_input_feature.iloc[test_index]
                Y_train, Y_test = train_input_label_list[train_index], train_input_label_list[test_index]
                auroc = roc_auc_score(Y_test, X_test)
                auprc = average_precision_score(Y_test, X_test)
                total_index_df.loc[len(total_index_df)] = [input_argue + '_valid', model_type, auroc, auprc]

        return total_index_df

    def get_performance_independent(self, model_type, label_list, index_list):
        total_index_df = pd.DataFrame(columns=['input_argue', 'model_type', 'auroc', 'auprc'])

        input_argue = 'combine'

        cossim_list = os.listdir('../result/cossim_csv')
        cossim_list.sort()
        for cv in range(10):
            type_info = [input_argue, self.subtype, str(cv), self.cancer_type, self.network_name, model_type]
            file_name = '_'.join(type_info) + '_meth.tsv'

            cos_sim_df = pd.read_csv('../result/cossim_csv/{}'.format(file_name), sep='\t')
            score_series = 1 - cos_sim_df['Tumor']

            # total label and score to performance
            auroc = roc_auc_score(label_list, score_series.iloc[index_list])
            auprc = average_precision_score(label_list, score_series.iloc[index_list])
            total_index_df.loc[len(total_index_df)] = [input_argue, model_type, auroc, auprc]

        return total_index_df

    def get_spearman_correlation(self, model_type, index_list):
        cossim_list = os.listdir('../result/cossim_csv')
        result = dict()
        for file_name in cossim_list:
            if file_name.endswith('{}.tsv'.format(model_type)):
                if model_type == 'network_100' and len(file_name.split('_')) == 6:
                    continue
                cos_sim_df = pd.read_csv('../result/cossim_csv/{}'.format(file_name), sep='\t')
                score_series = 1 - cos_sim_df['Tumor']
                score_series = score_series.iloc[index_list]
                result[file_name] = score_series

        spearman_df = pd.DataFrame(result)
        spearman_df = spearman_df.corr(method='spearman')
        return spearman_df


    def get_all_spearman_correlation(self, index_list):
        cossim_list = os.listdir('../result/cossim_csv')
        result = dict()
        for file_name in cossim_list:
            cos_sim_df = pd.read_csv('../result/cossim_csv/{}'.format(file_name), sep='\t')
            score_series = 1 - cos_sim_df['Tumor']
            score_series = score_series.iloc[index_list]
            file_name = file_name.split('.')[0][14:]
            file_name = file_name[2:] + '_' + file_name[0]
            result[file_name] = score_series

        spearman_df = pd.DataFrame(result)
        spearman_df = spearman_df.corr(method='spearman')
        spearman_df.sort_index(axis=0, inplace=True)
        spearman_df.sort_index(axis=1, inplace=True)
        return spearman_df

if __name__ == "__main__":
    ps = Performance_summary('BRCA', 'humannet_edges_FN')

    model_type_list = [#'full',
                       #'feature_only',
                       'structure_only',
                       # 'feature_25',
                       # 'feature_50',
                       # 'feature_75',
                       # 'feature_100',
                       # 'network_25',
                       # 'network_50',
                       # 'network_75',
                       # 'network_100',
                       # 'both_25',
                       # 'both_50',
                       # 'both_75',
                       # 'both_100',
                       # 'network-random_network_100',
                       # 'both-random_network_100'
                       ]

    ncg_symbol, oncokb_unique, negative_symbol = get_symbol_list(ps.cancer_type)
    oncokb_unique_index, ncg_index, total_negative_index, gene_list = get_symbol_index(ncg_symbol, oncokb_unique)

    test_index_list, test_label_list = ps.get_test_answer_index(ncg_symbol, oncokb_unique, negative_symbol)

    pos_only_train_index_list, pos_only_train_label_list, pos_only_test_index_list, pos_only_test_label_list = \
        get_pos_only_dataset(oncokb_unique_index, ncg_index, total_negative_index)

    random_train_index_list, random_train_label_list, random_test_index_list, random_test_label_list = \
        get_random_dataset(gene_list, negative_symbol, total_negative_index, ncg_index, oncokb_unique_index)
    train_index_list, train_label_list = ps.get_train_answer_index(ncg_symbol, negative_symbol)

    index_list, label_list = ps.get_total_answer_index(ncg_symbol, oncokb_unique, negative_symbol)
    positive_index_list, positive_label_list = ps.get_total_positive_answer_index(ncg_symbol, oncokb_unique)

    # result = get_performance_df(model_type_list, label_list, index_list)
    # result.to_csv("../result/performance/{}_{}_total_label_performance.tsv", sep="\t")

    # result = get_performance_df(model_type_list, train_label_list, train_index_list)
    # result.to_csv("../result/performance/{}_{}_train_label_performance.tsv".format(cancer_type, network_name), sep="\t")
    #
    # result = get_performance_df(model_type_list, test_label_list, test_index_list, True)
    # result.to_csv("../result/performance/{}_{}_test_independent_label_performance.tsv".format(cancer_type, network_name), sep="\t")

    result = ps.get_performance_df(model_type_list, pos_only_train_label_list, pos_only_train_index_list)
    result.to_csv("../result/performance/{}_{}_positive_label_performance.tsv".format(ps.cancer_type, ps.network_name), sep="\t")

    result = ps.get_performance_df(model_type_list, pos_only_test_label_list, pos_only_test_index_list, True)
    result.to_csv("../result/performance/{}_{}_test_positive_independent_label_performance.tsv".format(ps.cancer_type, ps.network_name), sep="\t")

    # result = get_performance_df(model_type_list, random_train_label_list, random_train_index_list)
    # result.to_csv("../result/performance/{}_{}_random_label_performance.tsv".format(cancer_type, network_name), sep="\t")
    #
    # result = get_performance_df(model_type_list, random_test_label_list, random_test_index_list, True)
    # result.to_csv("../result/performance/{}_{}_test_random_independent_label_performance.tsv".format(cancer_type, network_name), sep="\t")

    # total_spearman_df = get_all_spearman_correlation(index_list)
    # print(total_spearman_df)
    # total_spearman_df.to_csv("../result/analysis/spearman_corr.tsv", sep='\t')
