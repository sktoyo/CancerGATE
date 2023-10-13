import tensorflow as tf

class Grads:
    def __init__(self):
        self.mse = tf.keras.losses.MeanSquaredError()
        self.wce = tf.nn.weighted_cross_entropy_with_logits
        self.labels = None
        self.pos_weight = None
        self.norm = None

    def set_labels(self, labels):
        self.labels = labels

    def set_pos_weight(self, pos_weight):
        self.pos_weight = pos_weight

    def set_norm(self, norm):
        self.norm = norm

    def grad_multi(self, model, input_data, num_features):
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_weights)
            logits_st = model.get_reconstructed(input_data)
            loss_value_st = self.get_structure_loss(logits_st)
            # add feature loss batch
            logits_ft = model(input_data, training=True)
            loss_ft = tf.zeros(1)
            for j in range(num_features):
                loss_ft = tf.add(loss_ft, self.mse(input_data[j], logits_ft[j]))
            loss = tf.add(loss_ft, loss_value_st)
        grads = tape.gradient(loss, model.trainable_weights)
        return loss_ft, loss, grads

    def grad_single(self, model, input_data):
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_weights)
            logits_st = model.get_reconstructed(input_data)
            loss_value_st = self.get_structure_loss(logits_st)
            # add feature loss batch
            logits_ft = model(input_data, training=True)
            loss_ft = self.mse(input_data, logits_ft)
            loss = tf.add(loss_ft, loss_value_st)
        grads = tape.gradient(loss, model.trainable_weights)
        return loss_ft, loss, grads

    def grad_structure(self, model, input_data):
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_weights)
            logits_st = model.get_reconstructed(input_data)
            loss_value_st = self.get_structure_loss(logits_st)
            loss = loss_value_st
        grads = tape.gradient(loss, model.trainable_weights)
        return 0, loss, grads

    def get_structure_loss(self, logits_st):
        loss_value_st = tf.reduce_sum(self.wce(self.labels, logits_st, pos_weight=self.pos_weight))
        loss_value_st = loss_value_st / (self.labels.shape[0] ** 2) * self.norm
        return loss_value_st

    def get_feature_loss(self, input_data, logits_ft):
        return self.mse(input_data, logits_ft)
