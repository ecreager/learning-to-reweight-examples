import json
import numpy as np
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from datasets import adult
from utils import disparate_impact
from viz import curves

def train_meta_fair_classifier(
        train_loader,
        test_loader,
        n_epochs,
        learning_rate,
        layer_specs,
        lambda_fair,
        dirname='./fairness',
        eval_every=50):

    # build mlp model
    def _model(specs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        #layers = sum(
                #[[torch.nn.Linear(specs[i], specs[i+1]), torch.nn.ReLU()] for i in range(len(specs)-1)], 
                #[])
        #layers.pop()  # remove final nonlinearity; final layer should be linear
        from model import Simple
        model = Simple(specs)
        return model.to(device)

    def _model_and_opt(specs, lr):
        m = _model(specs)
        torch.backends.cudnn.benchmark=True
        #o = optim.Adam(list(m.parameters()), lr=lr)
        o = optim.Adam(list(m.params()), lr=lr)
        return m, o

    print('training fair LRE with lambda ', lambda_fair)
    hp = dict(n_epochs=n_epochs, learning_rate=learning_rate, layer_specs=layer_specs, lambda_fair=lambda_fair)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open('{}/opt.json'.format(dirname), 'w') as f:
        json.dump(hp, f)


    n_in = train_loader.dataset.train_data.shape[1]
    layer_specs.insert(0, n_in)

    _np = lambda t: t.detach().cpu().numpy()

    # get model
    net, opt = _model_and_opt(layer_specs, learning_rate)

    ## measure loss, accuracy, demographic parity for {train, test} every k iters
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    train_di_per_epoch = []
    test_loss_per_k_epochs = []
    test_acc_per_k_epochs = []
    test_di_per_k_epochs = []


    meta_losses_clean = []
    net_losses = []
    plot_step = 100

    meta_l = 0
    net_l = 0
    accuracy_log = []

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
                    y_logit = net(x).squeeze()
                    #_, y_hat = torch.max(y_logit, 1)
                    y_hat = y_logit > 0.
                    #di = disparate_impact(y_logit, a, train=False)
                    di = disparate_impact(y_logit, a, train=False)
                    loss = F.binary_cross_entropy_with_logits(y_logit, y.float()) + lambda_fair*di
                    loss_per_batch.append(_np(loss))
                    di_per_batch.append(di)
                    correct += (y_hat.int() == y.int()).sum().item()
                    total += len(y)
            test_accuracy = 100. * correct / total
            avg_loss = np.mean(loss_per_batch)
            test_loss_per_k_epochs.append(avg_loss)
            avg_di = np.mean(di_per_batch)
            test_di_per_k_epochs.append(avg_di)
            test_acc_per_k_epochs.append(test_accuracy)
            print('test', e, avg_loss, test_accuracy, avg_di)

        # train
        loss_per_batch = []
        meta_loss_per_batch = []
        di_per_batch = []
        meta_di_per_batch = []
        correct = 0
        total = 0
        for i, (x, a, y) in enumerate(train_loader):
            x, a, y = x.cuda(), a.cuda(), y.cuda()
            meta_net = _model(layer_specs)
            meta_net.load_state_dict(net.state_dict())

            if torch.cuda.is_available():
                meta_net.cuda()

            #y_f_logit  = meta_net(x).squeeze()
            y_f_logit  = meta_net(x)
            loss = F.binary_cross_entropy_with_logits(y_f_logit, y.float(), reduce=False)
            eps = Variable(torch.zeros(loss.size()), requires_grad=True).cuda()
            l_f_meta = torch.sum(loss * eps)

            meta_net.zero_grad()

            # Line 6 perform a parameter update
            #grads = torch.autograd.grad(l_f_meta, (meta_net.parameters()), create_graph=True)
            grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
            meta_net.update_params(learning_rate, source_params=grads)
            #for p, g in zip(list(meta_net.parameters()), grads):
            #    p = p - learning_rate*g

            # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
            valid_data = train_loader.dataset.valid_data.cuda()
            valid_attr = train_loader.dataset.valid_attr.cuda()
            valid_labels = train_loader.dataset.valid_labels.cuda()
            y_g_logit = meta_net(valid_data).squeeze()

            di = disparate_impact(y_g_logit, valid_attr, train=True)
            loss = F.binary_cross_entropy_with_logits(y_g_logit, valid_labels.float())
            l_g_meta = loss + lambda_fair*di
            #print(5, di)
            #print(6, loss)

            meta_loss_per_batch.append(l_g_meta.item())
            meta_di_per_batch.append(di.item())

            grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

            # Line 11 computing and normalizing the weights
            w_tilde = torch.clamp(-grad_eps,min=0)
            norm_c = torch.sum(w_tilde)

            if norm_c != 0:
                w = w_tilde / norm_c
            else:
                w = w_tilde

            # Lines 12 - 14 computing for the loss with the computed weights
            # and then perform a parameter update
            y_f_logit = net(x).squeeze()
            loss = F.binary_cross_entropy_with_logits(y_f_logit, y.float(), reduce=False)
            l_f = torch.sum(loss * w)

            opt.zero_grad()
            l_f.backward()
            opt.step()

            # TODO compute and store metrics
            loss_per_batch.append(l_f.item())
            y_hat = y_f_logit > 0.
            correct += (y_hat.int() == y.int()).sum().item()
            total += len(y)

        train_accuracy = 100. * correct / total
        avg_loss = np.mean(loss_per_batch)
        avg_meta_loss = np.mean(meta_loss_per_batch)
        avg_meta_di = np.mean(meta_di_per_batch)
        print('train', e, avg_loss, train_accuracy, 'm', avg_meta_loss, avg_meta_di)

    1/0

    return -1

if __name__ == '__main__':
    data_hyperparameters = dict(
            n_val=5,  # you get n_val from each (A, Y) combo so total valid set size is 4*n_val
            seed=None,
            batch_size=64,
            )
    train_loader, test_loader = adult(gold_std=True, **data_hyperparameters)

    # model hyperparams
    hyperparameters = dict(
        n_epochs=int(1.5e2),
        learning_rate=1e-3,
        layer_specs=[1],  # number of classes = 2
        #layer_specs=[8, 8, 8, 1],  # number of classes = 2
        #layer_specs=[8, 8, 8, 2],  # number of classes = 2
        #layer_specs=[2],  # number of classes = 2
        lambda_fair=1.,
        dirname='./fairness/meta_unfair_classifier',
        eval_every=10
        )
    if hyperparameters['lambda_fair'] > 0:
        hyperparameters.update(dirname='./fairness/meta_fair_classifier_{}'.format(hyperparameters['lambda_fair']))

    from pprint import pprint
    pprint(hyperparameters)
    print( train_meta_fair_classifier(train_loader, test_loader, **hyperparameters) )

1/0




