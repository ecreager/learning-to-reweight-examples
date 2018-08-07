import numpy as np


def curves(train, test, eval_every=50, basename='curves', dirname='./plots'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    f, a = plt.subplots(3)
    for i, (name, vals) in enumerate(zip(['train', 'test'], [train, test])):
        losses, accs, dis = vals
        idx = np.arange(len(losses))
        if name == 'test':
            idx *= eval_every
        a[0].plot(idx, losses, label=name)
        a[1].plot(idx, accs, label=name)
        a[2].plot(idx, dis, label=name)

    a[0].set_ylabel('loss')
    a[1].set_ylabel('acc')
    a[2].set_ylabel('disp imp')
    for aa in a:
        aa.legend()
        aa.set_xlabel('epoch')

    fn = '{}/{}.pdf'.format(dirname, basename)
    f.savefig(fn)
    plt.close(fn)
    return fn
    


if __name__ == '__main__':
    n_epochs = 500
    eval_every = 50
    
    # simulated metrics
    tr_l = np.random.randn(n_epochs)**2
    tr_acc = np.random.uniform(0.5, 1.0, n_epochs)
    tr_di = np.random.uniform(0.0, 0.2, n_epochs)

    te_l = tr_l[::eval_every]
    te_l += np.random.randn(*te_l.shape)
    te_acc = tr_acc[::eval_every]
    te_acc += np.random.randn(*te_acc.shape)
    te_di = tr_di[::eval_every]
    te_di += np.random.randn(*te_di.shape)

    train = list(tr_l), list(tr_acc), list(tr_di)
    test = list(te_l), list(te_acc), list(te_di)

    print(curves(train, test, eval_every, 'foo'))
