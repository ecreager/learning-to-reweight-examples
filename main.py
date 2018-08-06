
# coding: utf-8

# # Learning to Reweight Examples for Robust Deep Learning

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import *
from data_loader import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import matplotlib

hyperparameters = {
    'lr' : 1e-3,
    'momentum' : 0.9,
    'batch_size' : 100,
    'num_iterations' : 8000,
}


# ### Dataset
# Following the class imbalance experiment in the paper, we used numbers 9 and 4 of the MNIST dataset to form a highly imbalanced dataset where 9 is the dominating class. The test set on the other hand is balanced.

data_loader = get_mnist_loader(hyperparameters['batch_size'], classes=[9, 4], proportion=0.995, mode="train")
test_loader = get_mnist_loader(hyperparameters['batch_size'], classes=[9, 4], proportion=0.5, mode="test")

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


# #### Since the validation data is small (only 10 examples) there is no need to wrap it in a dataloader

val_data = to_var(data_loader.dataset.data_val, requires_grad=False)
val_labels = to_var(data_loader.dataset.labels_val, requires_grad=False)

for i,(img, label) in enumerate(data_loader):
    print(img.size(),label)
    break

for i,(img, label) in enumerate(test_loader):
    print(img.size(),label)
    break

def build_model():
    net = LeNet(n_out=1)

    if torch.cuda.is_available():
        net.cuda()
        torch.backends.cudnn.benchmark=True

    opt = torch.optim.SGD(net.params(),lr=hyperparameters["lr"])
    
    return net, opt


# ## Baseline Model
# I trained a LeNet model for the MNIST data without weighting the loss as a baseline model for comparison.

def train_baseline():
    print('training baseline')
    net, opt = build_model()

    net_losses = []
    plot_step = 100
    net_l = 0

    smoothing_alpha = 0.9
    accuracy_log = []
    for i in tqdm(range(hyperparameters['num_iterations'])):
        net.train()
        image, labels = next(iter(data_loader))

        image = to_var(image, requires_grad=False)
        labels = to_var(labels, requires_grad=False)

        y = net(image)
        cost = F.binary_cross_entropy_with_logits(y, labels)
        
        opt.zero_grad()
        cost.backward()
        opt.step()
        
        net_l = smoothing_alpha *net_l + (1 - smoothing_alpha)* cost.item()
        net_losses.append(net_l/(1 - smoothing_alpha**(i+1)))
        
        if i % plot_step == 0:
            net.eval()
            
            acc = []
            for itr,(test_img, test_label) in enumerate(test_loader):
                test_img = to_var(test_img, requires_grad=False)
                test_label = to_var(test_label, requires_grad=False)
                
                output = net(test_img)
                predicted = (torch.sigmoid(output) > 0.5).int()
                
                acc.append((predicted.int() == test_label.int()).float())

            accuracy = torch.cat(acc,dim=0).mean()
            accuracy_log.append(np.array([i,accuracy])[None])
            
            
    fig, axes = plt.subplots(1, 2, figsize=(13,5))
    ax1, ax2 = axes.ravel()

    ax1.plot(net_losses, label='net_losses')
    ax1.set_ylabel("Losses")
    ax1.set_xlabel("Iteration")
    ax1.legend()
    
    acc_log = np.concatenate(accuracy_log, axis=0)
    ax2.plot(acc_log[:,0],acc_log[:,1])
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Iteration')

    dn = './plots'
    if not os.path.exists(dn):
        os.makedirs(dn)
    fn = '{}/baseline.pdf'.format(dn)
    fig.savefig(fn)
    plt.close(fig)
    print('done training baseline; see plot at ')
    print(fn)



# UNCOMMENT TO TRAIN THE BASELINE
#print('baseline accuracy {}'.format(train_baseline()))

# As expected, due to the heavily imbalanced training data, the network could not learn how to differentiate between 9 and 4.

# ## Learning to Reweight Examples 
# Below is a pseudocode of the method proposed in the paper. It is very straightforward.

# <img src="pseudocode.PNG" width="300" />
# 
# 

# In[9]:


def train_lre(dl, repeat=0):  # dl is a data_loader from a call to get_mnist_loader
    p = data_loader.dataset.proportion
    val_data = to_var(dl.dataset.data_val, requires_grad=False)
    val_labels = to_var(dl.dataset.labels_val, requires_grad=False)

    dn = './plots'
    fn = '{}/lre-p{}-i{}.pdf'.format(dn, p, repeat)  # filename for plotting results


    print('training LRE with dataset imbalance {}'.format(p))

    net, opt = build_model()
    
    meta_losses_clean = []
    net_losses = []
    plot_step = 100

    smoothing_alpha = 0.9
    
    meta_l = 0
    net_l = 0
    accuracy_log = []
    for i in tqdm(range(hyperparameters['num_iterations'])):
        net.train()
        # Line 2 get batch of data
        image, labels = next(iter(dl))
        # since validation data is small I just fixed them instead of building an iterator
        # initialize a dummy network for the meta learning of the weights
        meta_net = LeNet(n_out=1)
        meta_net.load_state_dict(net.state_dict())

        if torch.cuda.is_available():
            meta_net.cuda()

        image = to_var(image, requires_grad=False)
        labels = to_var(labels, requires_grad=False)

        # Lines 4 - 5 initial forward pass to compute the initial weighted loss
        y_f_hat  = meta_net(image)
        cost = F.binary_cross_entropy_with_logits(y_f_hat,labels, reduce=False)
        eps = to_var(torch.zeros(cost.size()))
        l_f_meta = torch.sum(cost * eps)

        meta_net.zero_grad()
        
        # Line 6 perform a parameter update
        grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
        meta_net.update_params(hyperparameters['lr'], source_params=grads)
        
        # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
        y_g_hat = meta_net(val_data)

        l_g_meta = F.binary_cross_entropy_with_logits(y_g_hat,val_labels)

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
        y_f_hat = net(image)
        cost = F.binary_cross_entropy_with_logits(y_f_hat, labels, reduce=False)
        l_f = torch.sum(cost * w)

        opt.zero_grad()
        l_f.backward()
        opt.step()

        meta_l = smoothing_alpha *meta_l + (1 - smoothing_alpha)* l_g_meta.item()
        meta_losses_clean.append(meta_l/(1 - smoothing_alpha**(i+1)))

        net_l = smoothing_alpha *net_l + (1 - smoothing_alpha)* l_f.item()
        net_losses.append(net_l/(1 - smoothing_alpha**(i+1)))

        if i % plot_step == 0:
            net.eval()

            acc = []
            for itr,(test_img, test_label) in enumerate(test_loader):
                test_img = to_var(test_img, requires_grad=False)
                test_label = to_var(test_label, requires_grad=False)

                output = net(test_img)
                predicted = (torch.sigmoid(output) > 0.5).int()

                acc.append((predicted.int() == test_label.int()).float())

            accuracy = torch.cat(acc,dim=0).mean()
            accuracy_log.append(np.array([i,accuracy])[None])


            fig, axes = plt.subplots(1, 2, figsize=(13,5))
            ax1, ax2 = axes.ravel()

    ax1.plot(meta_losses_clean, label='meta_losses_clean')
    ax1.plot(net_losses, label='net_losses')
    ax1.set_ylabel("Losses")
    ax1.set_xlabel("Iteration")
    ax1.legend()

    acc_log = np.concatenate(accuracy_log, axis=0)
    ax2.plot(acc_log[:,0],acc_log[:,1])
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Iteration')

    if not os.path.exists(dn):
        os.makedirs(dn)
    fig.savefig(fn)
    plt.suptitle('LRE with p {} repeat {}'.format(p, repeat))
    plt.close(fig)
    print('done training LRE with p {} repeat {}; see plot at '.format(p, repeat))
    print(fn)

           
        # return accuracy
    return np.mean(acc_log[-6:-1, 1])


# To get an idea of how robust this method is with respect to the proportion of the dominant class, I varied the proportion from 0.9 to 0.995 and perform 5 runs for each. 

# In[10]:


num_repeats = 5
proportions = [0.9,0.95, 0.98, 0.99, 0.995]
accuracy_log = {}

for prop in proportions:
    data_loader = get_mnist_loader(hyperparameters['batch_size'], classes=[9, 4], proportion=prop, mode="train")
    
    for k in range(num_repeats):
        accuracy = train_lre(data_loader, k)
        
        if prop in accuracy_log:
            accuracy_log[prop].append(accuracy)
        else:
            accuracy_log[prop] = [accuracy]

plt.figure(figsize=(10,8))
for prop in proportions:
    accuracies = accuracy_log[prop]
    plt.scatter([prop] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(accuracy_log.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(accuracy_log.items())])
plt.errorbar(proportions, accuracies_mean, yerr=accuracies_std)
plt.title('Performance on varying class proportions')
plt.xlabel('proportions')
plt.ylabel('Accuracy')
plt.savefit('./plots/sweep.pdf')
#plt.show()


# We can see that even at 0.995 proportion of the dominant class in the training data, the model still reaches 90+% accuracy on the balanced test data.
