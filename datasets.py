# Elliot's dataset boilerplate stuff
# borrowed some code and ideas from the torchvision MNIST implementation
import os
import torch
import torch.utils.data as data
import torchvision


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
            self.train_labels = torch.tensor(np.argmax(dat['y_train'], axis=1), dtype=torch.int64)
            self.train_attr = torch.tensor(dat['attr_train'].squeeze(), dtype=torch.int64)  # sensitive attribute
            self.train_data = torch.tensor(dat['x_train'], dtype=torch.float32)
            if self.use_attr:
                self.train_data = torch.cat(
                        (self.train_data, self.train_attr.type(torch.float32)[:, None]), 
                        dim=1)
        else:
            self.test_labels = torch.tensor(np.argmax(dat['y_test'], axis=1), dtype=torch.int64)
            self.test_attr = torch.tensor(dat['attr_test'].squeeze(), dtype=torch.int64)  # sensitive attribute
            self.test_data = torch.tensor(dat['x_test'], dtype=torch.float32)
            if self.use_attr:
                self.test_data = torch.cat(
                        (self.test_data, self.test_attr.type(torch.float32)[:, None]), 
                        dim=1)

class UCIAdultGoldStd(UCIAdult):
    """
    like UCIAdult, except the training data is missing the sensitive attribute
    
    self.train_attr is still accessible but attr is omitted when iterated over
    """
    def __getitem__(self, index):
        inp, attr, target = super(UCIAdultGoldStd, self).__getitem__(index)
        if self.train:
            return inp, torch.tensor(float('nan')), target
        else:
            return inp, attr, target


def adult(batch_size, seed=None, gold_std=False):
    use_cuda = torch.cuda.is_available()
    if seed is not None:
        torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    dataset = UCIAdultGoldStd if gold_std else UCIAdult

    transform = None
    train_loader = torch.utils.data.DataLoader(
            dataset('./data', train=True, download=True, transform=transform, use_attr=False),  
            batch_size=batch_size, 
            shuffle=True, 
            **kwargs)

    test_loader = torch.utils.data.DataLoader(
            dataset('./data', train=False, download=True, transform=transform, use_attr=False),  
            batch_size=batch_size, 
            shuffle=True, 
            **kwargs)

    return train_loader, test_loader

if __name__ == '__main__':
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
