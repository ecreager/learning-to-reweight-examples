import numpy as np
import torch
import torch.optim as optim

from datasets import adult

# TODO
# * plot results
# * json stats
# * validation splits
# * compare baseline numbers for X->Y and X,A->Y


# SPECIFY WHETHER TO TRAIN ON SENSITIVE ATTR
## EITHER X -> Y OR X, A -> Y
USE_ATTR = False
N_EPOCHS = int(1e3)
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
N_OUT = 2  # number of classes
LAYERS = [8, 8, 8, N_OUT]

# MAKE UCI TRAIN AND TEST LOADERS
train_loader, test_loader = adult(BATCH_SIZE)

n_in = train_loader.dataset.train_data.shape[1]
LAYERS.insert(0, n_in)

# BUILD BASELINE MLP MODEL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
layers = sum(
        [[torch.nn.Linear(LAYERS[i], LAYERS[i+1]), torch.nn.ReLU()] for i in range(len(LAYERS)-1)], 
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

# INIT OPTIMIZER
optimizer = optim.Adam(list(model.parameters()), lr=LEARNING_RATE)

# TRAIN
## MEASURE LOSS, ACCURACY, DEMOGRAPHIC PARITY FOR {TRAIN, TEST} EVERY K ITERS
## WRITE TO JSON-FORMATTED LOG FILE
train_loss_per_epoch = []
train_acc_per_epoch = []
train_di_per_epoch = []
test_loss_per_k_epochs = []
test_acc_per_k_epochs = []
test_di_per_k_epochs = []
eval_every = 50

for e in range(N_EPOCHS):
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
                loss = loss_fn(y_logit, y)
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
        loss = loss_fn(y_logit, y)
        loss_per_batch.append(_np(loss))
        di_per_batch.append(di)
        correct += (y_hat == y).sum().item()
        total += len(y)
        
        # optimize
        loss.backward()
        optimizer.step()
    avg_loss = np.mean(loss_per_batch)
    avg_di = np.mean(di_per_batch)
    train_loss_per_epoch.append(avg_loss)
    train_acc_per_epoch.append(test_accuracy)
    train_accuracy = 100. * correct / total
    print('train', e, avg_loss, train_accuracy, avg_di)

# PLOT RESULT
