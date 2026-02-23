import os

import jittor.dataset as jt_dataset
from jittor.dataset import Dataset
import jittor.transform as transforms

ROOT = os.environ.get(
    "ATAS_DATA_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
)
shapes_dict = {'cifar10': [3, 40, 40], "cifar100": [3, 40, 40], "imagenet": [3, 32, 32]}
dir_dict = {'cifar10': "CIFAR10", 'cifar100': "CIFAR100", "imagenet": "imagenet"}
cls_dict = {'cifar10': 10, 'cifar100': 100, "imagenet": 1000}


def _get_dataset_cls(name):
    if hasattr(jt_dataset, name):
        return getattr(jt_dataset, name)

    if name in {'CIFAR10', 'CIFAR100'}:
        from jittor.dataset.cifar import CIFAR10, CIFAR100
        return {'CIFAR10': CIFAR10, 'CIFAR100': CIFAR100}[name]

    if name == 'ImageFolder':
        from jittor.dataset import ImageFolder
        return ImageFolder

    raise ImportError(f'Unsupported dataset class: {name}')


datasets_dict = {'cifar10': _get_dataset_cls('CIFAR10'), "cifar100": _get_dataset_cls('CIFAR100'), "imagenet": _get_dataset_cls('ImageFolder')}


class IndexDataset(Dataset):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, tuple):
            return item + (idx,)
        return item, idx

    def __len__(self):
        return len(self.dataset)


def load_data(dataset, batch_size, num_workers):
    assert dataset != 'imagenet'
    transform = transforms.Compose([transforms.Resize((40, 40)), transforms.ToTensor()])
    dir_ = os.path.join(ROOT, dir_dict[dataset])
    dataset_f = datasets_dict[dataset]

    train_data = dataset_f(root=dir_, train=True, transform=transform, download=True)
    train_data = IndexDataset(train_data)
    train_data.set_attrs(batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_data = dataset_f(root=dir_, train=False, transform=transform, download=True)
    test_data.set_attrs(batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_data, test_data, len(train_data)


def load_data_imagenet(dataset, batch_size, num_workers):
    assert dataset == 'imagenet'
    train_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor()])

    dir_ = os.path.join(ROOT, dir_dict[dataset])
    dataset_f = datasets_dict[dataset]

    train_data = dataset_f(root=os.path.join(dir_, "train"), transform=train_transform)
    train_data = IndexDataset(train_data)
    train_data.set_attrs(batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_data = dataset_f(root=os.path.join(dir_, "val"), transform=test_transform)
    test_data.set_attrs(batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_data, test_data, len(train_data)


def load_data_imagenet_val(dataset, batch_size, num_workers):
    assert dataset == 'imagenet'
    test_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor()])

    dir_ = os.path.join(ROOT, dir_dict[dataset])
    dataset_f = datasets_dict[dataset]

    test_data = dataset_f(root=os.path.join(dir_, "val"), transform=test_transform)
    test_data.set_attrs(batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_data
