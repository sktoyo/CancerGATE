from tqdm import tqdm


def train(model, input_data, optimizer, grad_tool, total_epoch=300):
    for epoch in tqdm(range(total_epoch)):
        loss_ft, loss, grads = grad_tool.grad_multi(model, input_data, len(input_data))
        optimizer.apply_gradients(
            (grad, var) for (grad, var) in zip(grads, model.trainable_variables) if
            grad is not None)  ### suppress warning


def train_single(model, input_data, optimizer, grad_tool, total_epoch=300):
    for epoch in tqdm(range(total_epoch)):
        loss_ft, loss, grads = grad_tool.grad_single(model, input_data)
        optimizer.apply_gradients(
            (grad, var) for (grad, var) in zip(grads, model.trainable_variables) if
            grad is not None)  ### suppress warning


def train_structure(model, input_data, optimizer, grad_tool, total_epoch=300):
    loss_list = list()
    for epoch in tqdm(range(total_epoch)):
        loss_ft, loss, grads = grad_tool.grad_structure(model, input_data)
        optimizer.apply_gradients(
            (grad, var) for (grad, var) in zip(grads, model.trainable_variables) if
            grad is not None)  ### suppress warning
        if epoch % 50 == 0:
            loss_list.append(loss)

    return loss_list