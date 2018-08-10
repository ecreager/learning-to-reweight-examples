import numpy as np
import os
import torch
from tqdm import tqdm

from datasets import adult


def train_guesser(train_loader, test_loader, model, lr=0.001, momentum=0.9, num_iterations=8000):
    """try to guess A from X using either logistic regression or a neural network"""

    assert model in ['neural_net', 'linear'], 'unsupported model'
    x_dim = train_loader.dataset.train_data.shape[1]
    n_classes = 2
    if model == 'linear':
        model = torch.nn.Linear(x_dim, n_classes)
    else:
        model = torch.nn.Sequential(
                torch.nn.Linear(x_dim, 500),
                torch.nn.ReLU(),
                torch.nn.Linear(500, 500),
                torch.nn.ReLU(),
                torch.nn.Linear(500, n_classes)
                )

    model = model.cuda()
    loss_fn = torch.nn.CrossEntropyLoss()

    opt = torch.optim.SGD(list(model.parameters()), lr=lr)

    net_losses = []
    plot_step = 100
    net_l = 0
    smoothing_alpha = 0.9
    accuracy_log = []
    for i in tqdm(range(num_iterations)):
        image, attr = train_loader.dataset.valid_data, train_loader.dataset.valid_attr
        image, attr = image.cuda(), attr.cuda()

        a = model(image)
        loss = loss_fn(a, attr)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        net_l = smoothing_alpha *net_l + (1 - smoothing_alpha)* loss.item()
        net_losses.append(net_l/(1 - smoothing_alpha**(i+1)))
        
        if i % plot_step == 0:
            
            acc = []
            for itr, (test_img, test_attr, _) in enumerate(test_loader):
                test_img, test_attr = test_img.cuda(), test_attr.cuda()
                
                output = model(test_img)
                predicted = torch.argmax(output, dim=1).int()
                
                acc.append((predicted.int() == test_attr.int()).float())

            accuracy = torch.cat(acc,dim=0).mean()
            accuracy_log.append(np.array([i,accuracy])[None])
            
            
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13,5))
    ax1, ax2 = axes.ravel()

    ax1.plot(net_losses, label='net_losses')
    ax1.set_ylabel("Losses")
    ax1.set_xlabel("Iteration")
    ax1.legend()

    acc_log = np.concatenate(accuracy_log, axis=0)
    ax2.plot(acc_log[:,0],acc_log[:,1])
    ax2.set_ylabel('Test Accuracy')
    ax2.set_xlabel('Iteration')

    dn = './plots'
    if not os.path.exists(dn):
        os.makedirs(dn)
    fn = '{}/guess_a--n_val-{}--model-{}.pdf'.format(dn, 
            train_loader.dataset.n_val,
            model)  # filename for plotting results
    fig.savefig(fn)
    plt.close(fig)
    print('done guessing a, see for yourself')
    print(fn)
    print('final accuracy of X -> A', acc_log[-1, 1] )

    # freeze model params; no more training from whatever downstream task
    for p in model.parameters():
        p.requires_grad = False
    return model

if __name__ == '__main__':
    hyperparameters = {
        'lr' : 1e-3,
        'momentum' : 0.9,
        #'num_iterations' : 8000,
        'num_iterations' : 16000,
        'n_val': 5,  # you get n_val from each (A, Y) combo so total valid set size is 4*n_val
        'model': 'linear',
        #'model': 'neural_net',
    }

    n_val = hyperparameters.pop('n_val')

    train_loader, test_loader = adult(128, gold_std=True, n_val=n_val)
    model = train_guesser(train_loader, test_loader, **hyperparameters)
    print(model)
    print(list(model.parameters()))
    for p in model.parameters():
        print(p.requires_grad)



