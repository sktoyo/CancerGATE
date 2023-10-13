import tensorflow as tf
from dgl.nn.tensorflow import GATConv


# class CancerGATE(tf.keras.Model):
#     def __init__(self,
#                  g,
#                  dim_hiddens,
#                  heads,
#                  activation,
#                  in_dims,
#                  feat_drop=0,
#                  attn_drop=0,
#                  negative_slope=0.2,
#                  residual=True,
#     ):
#         super(CancerGATE, self).__init__()
#         self.g = g
#         self.dim_list = dim_hiddens
#         self.embedding_layer_list = []
#         self.decoding_layer_list = []
#         self.layer_list = []
#         self.num_layers = len(dim_hiddens) - 1
#         self.num_features = len(in_dims)
#
#         # input embedding layers
#         for i in range(self.num_features):
#             self.embedding_layer_list.append(
#                 tf.keras.layers.Dense(self.dim_list[0], activation='relu', input_shape=(in_dims[i],)))
#
#         # output decoding layers
#         for i in range(self.num_features):
#             self.decoding_layer_list.append(tf.keras.layers.Dense(in_dims[i], activation='sigmoid'))
#
#         # hidden layers
#         for i in range(self.num_layers):
#             if i == 0:
#                 self.layer_list.append(GATConv(
#                     self.dim_list[i] * self.num_features, self.dim_list[i + 1], heads[i],
#                     feat_drop, attn_drop, negative_slope, residual, activation))
#             elif i != self.num_layers - 1:
#                 self.layer_list.append(GATConv(
#                     self.dim_list[i], self.dim_list[i + 1], heads[i],
#                     feat_drop, attn_drop, negative_slope, residual, activation))
#             # output layer
#             else:
#                 self.layer_list.append(GATConv(
#                     self.dim_list[i], self.dim_list[i + 1], heads[i],
#                     feat_drop, attn_drop, negative_slope, residual, activation=lambda x: x))
#
#         for i in range(self.num_layers, 0, -1):
#             # if i != 1:
#             self.layer_list.append(GATConv(
#                 self.dim_list[i], self.dim_list[i - 1], heads[i - 1],
#                 feat_drop, attn_drop, negative_slope, residual, activation))
#
#     def embedding(self, features):
#         """
#         Make embedding matrix of input features
#
#         param: 'features': input feature matrix
#
#         return: embedding matrix in logits
#         """
#         h = list()
#         for i, feature in enumerate(features):
#             h.append(self.embedding_layer_list[i](feature))
#
#         h = tf.keras.layers.Concatenate()(h)
#
#         for i, layer in enumerate(self.layer_list):
#             if i == self.num_layers:
#                 break
#             h = layer(self.g, h)
#             if i != self.num_layers - 1:
#                 h = tf.reshape(h, (h.shape[0], -1))
#         logits = tf.reduce_mean(h, axis=1)
#         return logits
#
#     def decoding(self, features):
#         """
#         Reconstruct the feature matrix
#
#         param 'features': input feature matrix
#
#         return: reconstructed feature matrix
#         """
#         h = features
#         h = tf.keras.activations.relu(h)
#         for i in range(self.num_layers, len(self.layer_list)):
#             h = self.layer_list[i](self.g, h)
#             if i != len(self.layer_list) - 1:
#                 h = tf.reshape(h, (h.shape[0], -1))
#
#         h = tf.reduce_mean(h, axis=1)
#
#         result = list()
#         for i in range(self.num_features):
#             result.append(self.decoding_layer_list[i](h))
#         return result
#
#     def get_reconstructed(self, features):
#         """
#         Reconstruct the adjacency matrix
#
#         param 'features': input feature matrix
#
#         return: reconstructed adjacency matrix
#         """
#         h = features
#         logits_h = self.embedding(h)
#         x = tf.transpose(logits_h)
#         logits_st = tf.matmul(logits_h, x)
#         return logits_st
#
#     def get_reconstructed_edge(self, features, edges):
#         """
#         Reconstruct the edges in probability
#
#         param 'features': input feature matrix
#
#         param 'edges': edge list
#
#         return:
#         """
#         h = features
#         logits_h = self.embedding(h)
#         edge_0 = [edge[0] for edge in edges]
#         edge_1 = [edge[1] for edge in edges]
#         logits_0 = tf.gather(logits_h, edge_0)
#         logits_1 = tf.gather(logits_h, edge_1)
#         logits_st = tf.math.multiply(logits_0, logits_1)
#         logits_st = tf.reduce_sum(logits_st, axis=1)
#         return logits_st
#
#     def get_attention(self, features):
#         """
#         Get attention values from input features
#
#         param 'features': input feature matrix
#
#         return: attention values in edge order
#         """
#         total_attention = 0
#         h = list()
#         for i, feature in enumerate(features):
#             h.append(self.embedding_layer_list[i](feature))
#
#         h = tf.keras.layers.Concatenate()(h)
#
#         for i, layer in enumerate(self.layer_list):
#             if i == self.num_layers:
#                 break
#             h, attention = layer(self.g, h, True)
#             if i != self.num_layers - 1:
#                 h = tf.reshape(h, (h.shape[0], -1))
#             attention = tf.reduce_sum(attention, axis=1)
#             total_attention += tf.reshape(attention, [attention.shape[0]])
#
#         return total_attention
#
#     def call(self, features):
#         """
#         Auto-encdoer part of feature
#
#         param 'features': input feature matrix
#
#         return: reconstructed feature matrix
#         """
#         h = features
#         h = self.embedding(h)
#         feature_re = self.decoding(h)
#
#         return feature_re


class CancerGATE(tf.keras.Model):
    def __init__(self,
                 g,
                 dim_hiddens,
                 heads,
                 activation,
                 in_dims,
                 feat_drop=0.2,
                 attn_drop=0.2,
                 negative_slope=0.2,
                 residual=True,
    ):
        super(CancerGATE, self).__init__()
        self.g = g
        self.dim_list = dim_hiddens
        self.embedding_layer_list = []
        self.decoding_layer_list = []
        self.layer_list = []
        self.num_layers = len(dim_hiddens) - 1
        self.num_features = len(in_dims)
        self.heads = heads

        # input embedding layers
        for i in range(self.num_features):
            self.embedding_layer_list.append(
                tf.keras.layers.Dense(self.dim_list[0], activation='relu', input_shape=(in_dims[i],)))

        # output decoding layers
        for i in range(self.num_features):
            self.decoding_layer_list.append(tf.keras.layers.Dense(in_dims[i], activation='sigmoid'))

        # hidden layers

        self.dropout = tf.keras.layers.Dropout(0.5)
        for i in range(self.num_layers):
            if i == 0:
                self.layer_list.append(GATConv(
                    self.dim_list[i] * self.num_features, self.dim_list[i + 1], heads[i],
                    feat_drop, attn_drop, negative_slope, residual, activation))
            elif i != self.num_layers - 1:
                self.layer_list.append(GATConv(
                    self.dim_list[i], self.dim_list[i + 1], heads[i],
                    feat_drop, attn_drop, negative_slope, residual, activation))
            # output layer
            else:
                self.layer_list.append(GATConv(
                    self.dim_list[i], self.dim_list[i + 1], heads[i],
                    feat_drop, attn_drop, negative_slope, residual, activation=lambda x: x))



    def embedding(self, features, training=True):
        """
        Make embedding matrix of input features

        param: 'features': input feature matrix

        return: embedding matrix in logits
        """
        h = list()
        for i, feature in enumerate(features):
            h.append(self.embedding_layer_list[i](feature))

        h = tf.keras.layers.Concatenate()(h)
        h = self.dropout(h, training=training)

        for i, layer in enumerate(self.layer_list):
            if i == self.num_layers:
                break
            h = layer(self.g, h, training=training)
            h = self.dropout(h, training=training)
            if i != self.num_layers - 1:
                h = tf.reshape(h, (h.shape[0], -1))
        return h

    def decoding(self, embedding_ft, training=True):
        """
        Reconstruct the feature matrix

        param 'features': input feature matrix

        return: reconstructed feature matrix
        """
        h = embedding_ft
        h = tf.keras.activations.relu(h)
        weights_list = list()
        for i, layer in enumerate(self.layer_list):
            weights_list.append(layer.fc._trainable_weights[0])
        for i in range(self.num_layers, 0, -1):
            if i == self.num_layers: # start of gcn weights
                h = tf.reshape(h, (h.shape[0], -1))
            h = tf.matmul(h, weights_list[i-1], transpose_b=True)
            h = tf.keras.activations.relu(h)
            if i == 1: # end of gcn weights
                h = tf.reshape(h, (h.shape[0], self.heads[0], -1))

        h = tf.reduce_mean(h, axis=1)

        result = list()
        for i in range(self.num_features):
            result.append(self.decoding_layer_list[i](h))
        return result

    def get_reconstructed(self, features, training=False):
        """
        Reconstruct the adjacency matrix

        param 'features': input feature matrix

        return: reconstructed adjacency matrix
        """
        h = features
        h = self.embedding(h, training)
        logits_h = tf.reduce_mean(h, axis=1)
        x = tf.transpose(logits_h)
        logits_st = tf.matmul(logits_h, x)
        return logits_st

    def get_reconstructed_edge(self, features, edges):
        """
        Reconstruct the edges in probability

        param 'features': input feature matrix

        param 'edges': edge list

        return:
        """
        h = features
        h = self.embedding(h, False)
        logits_h = tf.reduce_mean(h, axis=1)

        edge_0 = [edge[0] for edge in edges]
        edge_1 = [edge[1] for edge in edges]
        logits_0 = tf.gather(logits_h, edge_0)
        logits_1 = tf.gather(logits_h, edge_1)
        logits_st = tf.math.multiply(logits_0, logits_1)
        logits_st = tf.reduce_sum(logits_st, axis=1)
        return logits_st

    def get_attention(self, features):
        """
        Get attention values from input features

        param 'features': input feature matrix

        return: attention values in edge order
        """
        total_attention = 0
        h = list()
        for i, feature in enumerate(features):
            h.append(self.embedding_layer_list[i](feature))

        h = tf.keras.layers.Concatenate()(h)

        for i, layer in enumerate(self.layer_list):
            h, attention = layer(self.g, h, True, training=False)
            if i != self.num_layers - 1:
                h = tf.reshape(h, (h.shape[0], -1))
            attention = tf.reduce_sum(attention, axis=1)
            total_attention += tf.reshape(attention, [attention.shape[0]])

        return total_attention

    def call(self, features):
        """
        Auto-encdoer part of feature

        param 'features': input feature matrix

        return: reconstructed feature matrix
        """
        h = features
        h = self.embedding(h)
        feature_re = self.decoding(h)

        return feature_re
