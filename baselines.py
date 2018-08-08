import json
import numpy as np
import os
import torch
import torch.optim as optim

from datasets import adult
from viz import curves

# TODO
# * plot results
# * json stats
# * validation splits
# * compare baseline numbers for X->Y and X,A->Y


# specify whether to train on sensitive attr
## either x -> y or x, a -> y

def train_unfair_classifier(
        use_attr, 
        n_epochs,
        batch_size, 
        learning_rate,
        layer_specs,
        lambda_fair,
        dirname='./fairness'):

    hp = dict(use_attr=use_attr, n_epochs=n_epochs, batch_size=batch_size,
            learning_rate=learning_rate, layer_specs=layer_specs, lambda_fair=lambda_fair)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open('{}/opt.json'.format(dirname), 'w') as f:
        json.dump(hp, f)

    # make uci train and test loaders
    train_loader, test_loader = adult(batch_size)

    n_in = train_loader.dataset.train_data.shape[1]
    layer_specs.insert(0, n_in)

    # build baseline mlp model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    layers = sum(
            [[torch.nn.Linear(layer_specs[i], layer_specs[i+1]), torch.nn.ReLU()] for i in range(len(layer_specs)-1)], 
            [])
    layers.pop()  # remove final nonlinearity; final layer should be linear
    model = torch.nn.Sequential(*layers).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    _np = lambda t: t.detach().cpu().numpy()

    def _disparate_impact(y_hat, a):
        """disparate impact according to demographic parity"""
        return abs(torch.sub(
            y_hat[a == 0].type(torch.float32).mean(),
            y_hat[a == 1].type(torch.float32).mean()))

    # init optimizer
    optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)

    # train
    ## measure loss, accuracy, demographic parity for {train, test} every k iters
    ## write to json-formatted log file
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    train_di_per_epoch = []
    test_loss_per_k_epochs = []
    test_acc_per_k_epochs = []
    test_di_per_k_epochs = []
    eval_every = 50

    for e in range(n_epochs):
        if e % eval_every == 0:
            # evaluate loss, accuracy, fairness on test data
            loss_per_batch = []
            di_per_batch = []
            correct = 0
            total = 0
            with torch.no_grad():
                for i, (x, a, y) in enumerate(test_loader):
                    x, a, y = x.cuda(), a.cuda(), y.cuda()
                    n_group_0, n_group_1 = (1- a).sum(), a.sum()
                    y_logit = model(x)
                    _, y_hat = torch.max(y_logit, 1)
                    z = n_group_0.type(torch.float32)
                    di = _disparate_impact(y_hat, a)
                    loss = loss_fn(y_logit, y) + lambda_fair*_disparate_impact(torch.sigmoid(y_logit), a)
                    loss_per_batch.append(_np(loss))
                    di_per_batch.append(di)
                    correct += (y_hat == y).sum().item()
                    total += len(y)
            test_accuracy = 100. * correct / total
            avg_loss = np.mean(loss_per_batch)
            test_loss_per_k_epochs.append(avg_loss)
            avg_di = np.mean(di_per_batch)
            test_di_per_k_epochs.append(avg_di)
            test_acc_per_k_epochs.append(test_accuracy)
            print('test', e, avg_loss, test_accuracy, avg_di)

        loss_per_batch = []
        correct = 0
        total = 0
        for i, (x, a, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x, a, y = x.cuda(), a.cuda(), y.cuda()
            y_logit = model(x)
            _, y_hat = torch.max(y_logit, 1)
            di = _disparate_impact(y_hat, a)
            loss = loss_fn(y_logit, y) + lambda_fair*_disparate_impact(torch.sigmoid(y_logit), a)
            loss_per_batch.append(_np(loss))
            di_per_batch.append(di)
            correct += (y_hat == y).sum().item()
            total += len(y)

            # optimize
            loss.backward()
            optimizer.step()

        # compute avg metrics per epoch
        train_accuracy = 100. * correct / total
        avg_loss = np.mean(loss_per_batch)
        avg_di = np.mean(di_per_batch)
        train_loss_per_epoch.append(avg_loss)
        train_acc_per_epoch.append(train_accuracy)
        train_di_per_epoch.append(avg_di)
        train_accuracy = 100. * correct / total
        print('train', e, avg_loss, train_accuracy, avg_di)

    # save metrics from final iteration
    final_metrics = dict(
            train_loss=train_loss_per_epoch[-1].item(),
            train_acc=train_acc_per_epoch[-1],
            train_di=train_di_per_epoch[-1].item(),
            test_loss=test_loss_per_k_epochs[-1].item(),
            test_acc=test_acc_per_k_epochs[-1],
            test_di=test_di_per_k_epochs[-1].item())

    with open('{}/final_metrics.json'.format(dirname), 'w') as f:
        json.dump(final_metrics, f)
        print('saved final metrics to disk at', f.name)

    # plot result
    return(curves(
        (train_loss_per_epoch, train_acc_per_epoch, train_di_per_epoch),
        (test_loss_per_k_epochs, test_acc_per_k_epochs, test_di_per_k_epochs),
        eval_every,
        dirname=dirname))


if __name__ == '__main__':
    from collections import OrderedDict
    hyperparameters = OrderedDict(
        use_attr=False,
        n_epochs=int(1e3),
        batch_size=64,
        learning_rate=1e-3,
        layer_specs=[8, 8, 8, 2],  # number of classes = 2
        lambda_fair=50.,
        dirname='./fairness/unfair_classifier'
        )
    if hyperparameters['lambda_fair'] > 0:
        hyperparameters.update(dirname='./fairness/regularized_fair_classifier_{}'.format(hyperparameters['lambda_fair']))

    print( train_unfair_classifier(**hyperparameters) )

1/0
