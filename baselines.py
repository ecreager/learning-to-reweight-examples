import json
import numpy as np
import os
import torch
import torch.optim as optim

from datasets import adult
from utils import disparate_impact
from viz import curves

# TODO
# * json stats
# * compare baseline numbers for X->Y and X,A->Y


def train_fair_classifier(
        train_loader,
        test_loader,
        n_epochs,
        learning_rate,
        layer_specs,
        lambda_fair,
        guesser=None,
        dirname='./fairness',
        eval_every=50):

    hp = dict(n_epochs=n_epochs, 
            learning_rate=learning_rate, layer_specs=layer_specs, lambda_fair=lambda_fair, guesser=str(guesser))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open('{}/opt.json'.format(dirname), 'w') as f:
        json.dump(hp, f)

    if guesser is not None:
        assert not train_loader.dataset.use_attr, 'cant have A in X when using a guesser'

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

    # init optimizer
    optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)

    ## measure loss, accuracy, demographic parity for {train, test} every k iters
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    train_di_per_epoch = []
    valid_loss_per_k_epochs = []
    valid_acc_per_k_epochs = []
    valid_di_per_k_epochs = []
    test_loss_per_k_epochs = []
    test_acc_per_k_epochs = []
    test_di_per_k_epochs = []

    for e in range(n_epochs):
        if e % eval_every == 0:
            # evaluate loss, accuracy, fairness on test data
            loss_per_batch = []
            di_per_batch = []
            correct = 0
            total = 0
            with torch.no_grad():
                # test metrics
                for i, (x, a, y) in enumerate(test_loader):
                    x, a, y = x.cuda(), a.cuda(), y.cuda()
                    y_logit = model(x)
                    _, y_hat = torch.max(y_logit, 1)
                    p_y0 = torch.sigmoid(y_logit)[:, 0]
                    di = disparate_impact(y_logit, a, train=False)
                    loss = loss_fn(y_logit, y) + lambda_fair*di
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
                # validation metrics
                x, a, y = train_loader.dataset.valid_data, train_loader.dataset.valid_attr, train_loader.dataset.valid_attr
                x, a, y = x.cuda(), a.cuda(), y.cuda()
                y_logit = model(x)
                valid_di = disparate_impact(y_logit, a, train=False)
                valid_loss = loss_fn(y_logit, y) + lambda_fair*valid_di
                valid_accuracy = 100. * (y_hat == y).sum().item() / len(y)
                valid_di, valid_loss = valid_di.item(), valid_loss.item()
                valid_loss_per_k_epochs.append(valid_loss)
                valid_di_per_k_epochs.append(valid_di)
                valid_acc_per_k_epochs.append(valid_accuracy)
                print('valid', e, valid_loss, valid_accuracy, valid_di)

        # train
        loss_per_batch = []
        di_per_batch = []
        if guesser is not None:
            valid_di_per_batch = []
        correct = 0
        total = 0
        for i, (x, a, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x, a, y = x.cuda(), a.cuda(), y.cuda()
            if guesser is not None:  # guessing the sensitive attribute
                a = guesser(x).argmax(dim=1)
            y_logit = model(x)
            _, y_hat = torch.max(y_logit, 1)
            di = disparate_impact(y_logit, a, train=True)
            loss = loss_fn(y_logit, y) + lambda_fair*di
            loss_per_batch.append(_np(loss))
            di_per_batch.append(_np(di))
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
        # validation metrics
        if guesser is not None:  # how are we doing on the data we used to train X -> A?
            _, ahat = torch.max(guesser(train_loader.dataset.valid_data.cuda()), 1)
            _, yhat = torch.max(model(train_loader.dataset.valid_data.cuda()), 1)
            valid_di = disparate_impact(model(train_loader.dataset.valid_data.cuda()), train_loader.dataset.valid_attr.cuda(), train=True)
            valid_loss = loss_fn(
                    model(train_loader.dataset.valid_data.cuda()), 
                    train_loader.dataset.valid_labels.cuda()) + lambda_fair*valid_di
            valid_di, valid_loss = valid_di.item(), valid_loss.item()
        print('train', e, avg_loss, train_accuracy, avg_di, 
                'v' if guesser is not None else '',
                valid_loss if guesser is not None else '',
                valid_di if guesser is not None else '',
                )
        #print('train', e, avg_loss, train_accuracy, avg_di)

    # save metrics from final iteration
    final_metrics = dict(
            train_loss=train_loss_per_epoch[-1].item(),
            train_acc=train_acc_per_epoch[-1],
            train_di=train_di_per_epoch[-1].item(),
            valid_loss=valid_loss_per_k_epochs[-1],
            valid_acc=valid_acc_per_k_epochs[-1],
            valid_di=valid_di_per_k_epochs[-1],
            test_loss=test_loss_per_k_epochs[-1].item(),
            test_acc=test_acc_per_k_epochs[-1],
            test_di=test_di_per_k_epochs[-1].item())

    with open('{}/final_metrics.json'.format(dirname), 'w') as f:
        json.dump(final_metrics, f)
        print('saved final metrics to disk at', f.name)

    # plot result
    return(curves(
        (train_loss_per_epoch, train_acc_per_epoch, train_di_per_epoch),
        (valid_loss_per_k_epochs, valid_acc_per_k_epochs, valid_di_per_k_epochs),
        (test_loss_per_k_epochs, test_acc_per_k_epochs, test_di_per_k_epochs),
        eval_every,
        dirname=dirname,
        ylim=[50., 100.]))


if __name__ == '__main__':
    data_hyperparameters = dict(
            n_val=5,  # you get n_val from each (A, Y) combo so total valid set size is 4*n_val
            seed=None,
            batch_size=64,
            )
    GOLD_STD = True
    train_loader, test_loader = adult(gold_std=GOLD_STD, **data_hyperparameters)

    # model hyperparams
    hyperparameters = dict(
        n_epochs=int(1.e2),
        learning_rate=1e-3,
        layer_specs=[8, 8, 8, 2],  # number of classes = 2
        #layer_specs=[2],  # number of classes = 2
        lambda_fair=1.,
        dirname='./fairness/unfair_classifier',
        guess=GOLD_STD,
        eval_every=10
        )
    if hyperparameters['lambda_fair'] > 0:
        hyperparameters.update(dirname='./fairness/regularized_fair_classifier_{}'.format(hyperparameters['lambda_fair']))

    if hyperparameters.pop('guess'):
        print('training guesser X -> A using gold std data')
        hyperparameters['dirname'] += '_guesser-nv{}'.format(data_hyperparameters['n_val'])
        from guess_a import train_guesser
        guesser_hyperparameters = dict(
                lr=1e-3,
                momentum=0.9,
                num_iterations=8000,
                model='linear')
        hyperparameters['guesser'] = train_guesser(train_loader, test_loader, **guesser_hyperparameters)

    from pprint import pprint
    pprint(hyperparameters)
    print( train_fair_classifier(train_loader, test_loader, **hyperparameters) )

1/0
