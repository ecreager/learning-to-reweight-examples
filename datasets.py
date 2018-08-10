# Elliot's dataset boilerplate stuff
# borrowed some code and ideas from the torchvision MNIST implementation
import os
import torch
import torch.utils.data as data
import torchvision

from viz import grid


class UCIAdult(data.Dataset):
    """UCI Adult dataset: https://archive.ics.uci.edu/ml/datasets/adult
    
    this implementation assumes that the pre-processed npz files exists as ./data/adult.npz
    """
    data_filename = './data/adult.npz'
    classes = ['0, income < 50k', '1, income >= 50k']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    @property
    def targets(self):
        return self.train_labels if self.train else self.test_labels

    def __init__(self, root, train=True, transform=None, target_transform=None, attr_transform=None, download=False, use_attr=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.attr_transform = attr_transform
        self.train = train  # training set or test set
        self.use_attr = use_attr  # whether or not inputs X contain sensitive attribute A

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                    ' You can use download=True to download it')

        self.load_from_disk()

    def __getitem__(self, index):
        """
        Args:
        index (int): Index

        Returns:
        tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            inp, attr, target = self.train_data[index], self.train_attr[index], self.train_labels[index]
        else:
            inp, attr, target = self.test_data[index], self.test_attr[index], self.test_labels[index]

        #inp = inp.numpy()  # make ammenable to torchvision transforms

        if self.transform is not None:
            inp = self.transform(inp)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # TODO: implement a gold standard attribute transform
        # it all attr=None in the test set and splits a validation set
        # this could alternatively be implemented in a sub or superlcass of UCIAdult
        if self.attr_transform is not None:  
            target = self.attr_transform(attr)

        return inp, attr, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(self.data_filename)

    def load_from_disk(self):
        import numpy as np
        dat = dict(np.load(self.data_filename))

        if self.train:
            self.train_labels = torch.tensor(np.argmax(dat['y_train'], axis=1), dtype=torch.long)
            self.train_attr = torch.tensor(dat['attr_train'].squeeze(), dtype=torch.long)  # sensitive attribute
            self.train_data = torch.tensor(dat['x_train'], dtype=torch.float)
            if self.use_attr:
                self.train_data = torch.cat(
                        (self.train_data, self.train_attr.type(torch.float)[:, None]), 
                        dim=1)
        else:
            self.test_labels = torch.tensor(np.argmax(dat['y_test'], axis=1), dtype=torch.long)
            self.test_attr = torch.tensor(dat['attr_test'].squeeze(), dtype=torch.long)  # sensitive attribute
            self.test_data = torch.tensor(dat['x_test'], dtype=torch.float)
            if self.use_attr:
                self.test_data = torch.cat(
                        (self.test_data, self.test_attr.type(torch.float)[:, None]), 
                        dim=1)


class UCIAdultGoldStd(UCIAdult):
    """
    like UCIAdult, except the training data is missing the sensitive attribute
    self.train_attr is still accessible but attr is omitted when iterated over
    self.valid_* contain the gold std validation set
    """
    def __init__(self, root, *args, n_val=5, **kwargs):
        if 'use_attr' in kwargs.keys():
            kwargs.update(use_attr=False)
        super(UCIAdultGoldStd, self).__init__(root, *args, **kwargs)
        self.n_val = n_val
        # build "gold std" validation set of n_val examples from each (A, Y) combination
        if self.train:
            data, attr, labels = self.train_data, self.train_attr, self.train_labels
        else:
            data, attr, labels = self.test_data, self.test_attr, self.test_labels

        # map (A, Y) to indices where a=A, y=Y
        inds = {(i, j): (attr == i) * (labels == j) for i in range(2) for j in range(2)} 

        # data, attr, label for each subgroup
        xIay = {k: data[v, :] for k, v in inds.items()}
        aIay = {k: attr[v] for k, v in inds.items()}
        yIay = {k: attr[v] for k, v in inds.items()}
        for k in inds.keys():
            assert len(xIay[k]) >= n_val, 'too few examples to make validation set'
            assert len(aIay[k]) >= n_val, 'too few examples to make validation set'
            assert len(yIay[k]) >= n_val, 'too few examples to make validation set'
            xIay[k] = xIay[k][:n_val, :]
            aIay[k] = aIay[k][:n_val]
            yIay[k] = yIay[k][:n_val]
        self.valid_data = torch.cat([v for v in xIay.values()], dim=0)
        self.valid_attr = torch.cat([v for v in aIay.values()], dim=0)
        self.valid_labels = torch.cat([v for v in yIay.values()], dim=0)

    def __getitem__(self, index):
        inp, attr, target = super(UCIAdultGoldStd, self).__getitem__(index)
        if self.train:
            return inp, torch.tensor(float('nan')), target
        else:
            return inp, attr, target


def adult(batch_size=64, seed=None, gold_std=False, n_val=5):
    use_cuda = torch.cuda.is_available()
    if seed is not None:
        torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    dataset = UCIAdultGoldStd if gold_std else UCIAdult
    dataset_kwargs = {'n_val': n_val} if gold_std else {}

    transform = None
    train_loader = torch.utils.data.DataLoader(
            dataset('./data', train=True, download=True, transform=transform, use_attr=False, **dataset_kwargs),  
            batch_size=batch_size, 
            shuffle=True, 
            **kwargs)

    test_loader = torch.utils.data.DataLoader(
            dataset('./data', train=False, download=True, transform=transform, use_attr=False, **dataset_kwargs),  
            batch_size=batch_size, 
            shuffle=True, 
            **kwargs)

    return train_loader, test_loader


class CelebA(torchvision.datasets.ImageFolder):
    celeba_dirname  = '/scratch/gobi1/datasets/celeb-a'
    eval_partition_filename = '/scratch/gobi1/datasets/celeb-a/list_eval_partition.csv'
    attr_filename = '/scratch/gobi1/datasets/celeb-a/list_attr_celeba.csv'

    def __init__(self, train=True, **kwargs):
        from datetime import datetime
        import pandas as pd
        t = datetime.now()
        print('loading CelebA data')
        super(CelebA, self).__init__(self.celeba_dirname, **kwargs)
        print('done; it took {} secs'.format(datetime.now() - t))
        self.eval_partition = pd.read_csv(self.eval_partition_filename)
        self.attr = pd.read_csv(self.attr_filename)
        self.attr = self.attr * (self.attr > 0)  # {-1, 1} -> {0, 1}
        self.train = train  # training set or test set; NB validation set currently not supported

        self.train_range = self._get_range(self.eval_partition, 0)
        self.valid_range = self._get_range(self.eval_partition, 1)
        self.test_range = self._get_range(self.eval_partition, 2)

        if self.train:
            del self.samples[self.test_range[0]:self.test_range[1]]
            del self.imgs[self.test_range[0]:self.test_range[1]]
            del self.samples[self.valid_range[0]:self.valid_range[1]]
            del self.imgs[self.valid_range[0]:self.valid_range[1]]
            self.attr = self.attr.loc[self.train_range[0]:self.train_range[1]].reset_index()
        else:
            del self.samples[self.train_range[0]:self.valid_range[1]]
            del self.imgs[self.train_range[0]:self.valid_range[1]]
            self.attr = self.attr.loc[self.test_range[0]:self.test_range[1]].reset_index()

        from collections import OrderedDict
        self.idx_to_attr = OrderedDict({i: self.attr.columns[i+2] for i in range(40)})

    def __getitem__(self, index):
        img, fake_label = super(CelebA, self).__getitem__(index)
        attrs = [self.attr[self.idx_to_attr[j]].__getitem__(index) for j in range(40)]
        return img, torch.tensor(attrs, dtype=torch.long)

    @staticmethod
    def _get_range(df, partition_val):  
        """partition val in {0, 1, 2} represents {train, valid, test}"""
        min_idx = df['partition'][df['partition'] == partition_val].index.min()
        max_idx = df['partition'][df['partition'] == partition_val].index.max()
        return min_idx, max_idx

    def _attrs_to_str(self, list_of_int_attrs):
        s = ''
        for i in range(40):
            s += '\n{} = {}'.format(self.idx_to_attr[i], list_of_int_attrs[i])
        return s


def celeba(batch_size, seed=None):  # TODO: CelebAGoldStd
    use_cuda = torch.cuda.is_available()
    if seed is not None:
        torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    #transform = None
    transform = torchvision.transforms.Compose([
        #torchvision.transforms.Grayscale(),  
        #transforms.Lambda(lambda x: x/255.),
        torchvision.transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
            CelebA(train=True, transform=transform),  
            batch_size=batch_size, 
            shuffle=True, 
            **kwargs)

    test_loader = torch.utils.data.DataLoader(
            CelebA(train=False, transform=transform),  
            batch_size=batch_size, 
            shuffle=True, 
            **kwargs)

    return train_loader, test_loader


if __name__ == '__main__':
    if True:  # test uci adult
        GOLD_STD = True

        train_loader, test_loader = adult(128, gold_std=GOLD_STD)
        print('gold standard uci adult dataset' if GOLD_STD else 'uci adult dataset')

        for loader in [train_loader, test_loader]:
            print()
            print('training data' if loader.dataset.train else 'test data')
            for batch_idx, (x, a, y) in enumerate(loader):
                if batch_idx >= 1:
                    break
                x, a, y = x.cuda(), a.cuda(), y.cuda()
                print('x', x)
                print('a', a)
                print('y', y)
                print('x a y shapes', x.shape, a.shape, y.shape)

        print('done')
    else:  # test celeba
        train_loader, test_loader = celeba(128)
        for loader in [train_loader, test_loader]:
            print()
            print('training data' if loader.dataset.train else 'test data')
            for batch_idx, (x, y) in enumerate(loader):
                print(x.__class__, y.__class__)
                if batch_idx >= 1:
                    break
                x, y = x.cuda(), y.cuda()
                print('x', x)
                print('y', y)
                print('x y shapes', x.shape, y.shape)
                name = 'celeba-train' if loader.dataset.train else 'celeba-test'
                print(grid(x[:16], name))

          
