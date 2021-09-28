import os
from os.path import join
import sys
from torch.utils import data
import torchvision.transforms as transforms
from . import transforms as transforms_custom
import numpy as np


__all__ = names = ("WFUndefended", "WFUndefendedOW", "WFSpring", "WFSpringOW", "WFSubpages24")


class NoneTransform(object):
    """ Does nothing to the image, to be used instead of None

    Args:
        image in, image out, nothing is done
    """
    def __call__(self, image):
        return image


class Pylls(data.Dataset):
    def __init__(self, root="./data", *args, train=True, download=False, **kwargs):
        super().__init__()

        if train:
            dataset, labels, classes, ids = load_dataset(
                    join(root, 'mon'), join(root, 'unmon'), join(root, 'classes.list'), 
                    samples = list(range(0,18)), unm_partition = 0,
                    **kwargs)
        else:
            dataset, labels, classes, ids = load_dataset(
                    join(root, 'mon'), join(root, 'unmon'), join(root, 'classes.list'), 
                    samples = list(range(18,20)), unm_partition = 9000,
                    **kwargs)
        self.classes = classes
        self.ids = ids
        self.dataset = dataset
        self.labels = labels
 
    def __len__(self):
        return len(self.ids)
 
    def __getitem__(self, index):
        ID = self.ids[index]
        return self.dataset[ID], self.labels[ID]

    @staticmethod
    def transform_train(input_size=64):
        return transforms.Compose([NoneTransform()])

    @staticmethod
    def transform_val(input_size=-1):
        return transforms.Compose([NoneTransform()])

    @staticmethod
    def transform_val_inverse():
        return transforms_custom.InverseNormalizeNone()


class Subpages(data.Dataset):
    def __init__(self, root="./data", *args, train=True, download=False, **kwargs):
        super().__init__()

        if train:
            dataset, labels, classes, ids = load_dataset_subpages(
                    root,
                    samples = list(range(0,80)),
                    **kwargs)
        else:
            dataset, labels, classes, ids = load_dataset_subpages(
                    root,
                    samples = list(range(80,90)),
                    **kwargs)
        self.classes = classes
        self.ids = ids
        self.dataset = dataset
        self.labels = labels
 
    def __len__(self):
        return len(self.ids)
 
    def __getitem__(self, index):
        ID = self.ids[index]
        return self.dataset[ID], self.labels[ID]

    @staticmethod
    def transform_train(input_size=64):
        return transforms.Compose([NoneTransform()])

    @staticmethod
    def transform_val(input_size=-1):
        return transforms.Compose([NoneTransform()])

    @staticmethod
    def transform_val_inverse():
        return transforms_custom.InverseNormalizeNone()


def load_trace(
        fi, separator='\t', filter_acks = True
        ):
    sample = [[],[],[]]
    for line in fi:
        ts, dsize = line.split(separator)
        ts = float(ts)
        dsize = int(dsize)
        if filter_acks and abs(dsize) < 512:
            continue
        sample[0].append(ts)
        sample[1].append(abs(dsize))
        sample[2].append(dsize//abs(dsize))
    return sample


def process(
        sample, length
        ):
    proc = []
    try:
        for i in range(1,len(sample[0])):
            iat = sample[0][i] - sample[0][i-1]
            val = (iat + 1.) * sample[2][i]
            val = round(val,3)
            proc.append(val)
    except Exception as e:
        print('Failed processing trace with error,', e)
    proc = np.array(proc)
    if len(proc) < length:
        proc = np.pad(proc, (0,length-len(proc)))
    elif len(proc) > length:
        proc = proc[:length]
    return proc


def load_dataset(
    mon_dir, unm_dir, classes_path,
    classes = 50, subpages = 10,
    samples = list(range(20)), 
    unm_partition = 0,
    length = 7000, 
    subpage_as_label = False,
    include_unm = False,
    defen_multiples = 0,
    **kwargs
    ):
    ''' Loads the dataset from disk into two dictionaries for data and labels.
    The dictionaries are indexed by sample ID. The ID encodes if its a monitored
    or unmonitored sample to make it easier to debug, as well as some info about
    the corresponding data file on disk. 
    This function works assumes the structure of the following dataset:
    - "top50-partitioned-reddit-levels-cirucitpadding" 
    '''
    data = {}
    labels = {}
    IDs = []

    with open(classes_path) as fi:
        if subpage_as_label:
            class_names = ['{line}-{i}' for i in range(partitions) for line in fi]
        else:
            class_names = [line for line in fi]

    # load monitored data
    for c in range(0,classes):
        for p in range(0,subpages):
            site = c*subpages + p
            for s in samples:
                if defen_multiples <= 0:
                    ID = f"m-{c}-{p}-{s}"
                    if subpage_as_label:
                        labels[ID] = site
                    else:
                        labels[ID] = c
                    IDs.append(ID)

                    # file format is {site}-{sample}.trace
                    fname = f"{site}-{s}"
                    with open(join(mon_dir, fname), "r") as f:
                        data[ID] = process(load_trace(f), length)
                else:
                    for m in range(defen_multiples):
                        ID = f"m-{c}-{p}-{s}-{m}"
                        if subpage_as_label:
                            labels[ID] = site
                        else:
                            labels[ID] = c
                        IDs.append(ID)

                        # file format is {site}-{sample}.trace
                        fname = f"{site}-{s}-{m}"
                        with open(join(mon_dir, fname), "r") as f:
                            data[ID] = process(load_trace(f), length)


    if include_unm:
        # load unmonitored data
        dirlist = sorted(os.listdir(unm_dir))
        # make sure we only load a balanced dataset
        dirlist = dirlist[unm_partition:len(data)]
        for fname in dirlist:
            ID = f"u-{fname}"
            labels[ID] = classes # start from 0 for monitored
            with open(os.path.join(unm_dir, fname), "r") as f:
                data[ID] = process(load_trace(f), length)
        class_names.append('unmonitored')

    return data, labels, class_names, IDs


def load_dataset_subpages(
    mon_dir,
    subpages = 90,
    samples = list(range(90)), 
    length = 7000, 
    subpage_as_label = False,
    **kwargs
    ):
    '''
    '''
    data = {}
    labels = {}
    IDs = []

    class_names = []

    dirlist = sorted(os.listdir(mon_dir))
    for c,dirname in enumerate(dirlist):
        if not subpage_as_label:
            class_names.append(dirname)
        for p in range(0,subpages):
            if subpage_as_label:
                class_names.append(f'{dirname}-{p}')
            for s in samples:
                ID = f"m-{dirname}-{p}-{s}"
                if subpage_as_label:
                    labels[ID] = c*subpages + p
                else:
                    labels[ID] = c
                IDs.append(ID)

                # file format is {site}-{sample}.trace
                fname = f"{p}-{s}"
                with open(join(mon_dir, dirname, fname), "r") as f:
                    data[ID] = process(load_trace(f), length)

    return data, labels, class_names, IDs


class WFUndefended(Pylls):
    def __init__(self, root, *args, **kwargs):
        super().__init__(join(root, 'wf-undefended'), *args, 
                length=7000, **kwargs)


class WFUndefendedOW(Pylls):
    def __init__(self, root, *args, **kwargs):
        super().__init__(join(root, 'wf-undefended'), *args, 
                length=7000, include_unm=True, **kwargs)


class WFSpring(Pylls):
    def __init__(self, root, *args, **kwargs):
        super().__init__(join(root, 'wf-spring'), *args, 
                length=9000, defen_multiples=20, **kwargs)


class WFSpringOW(Pylls):
    def __init__(self, root, *args, **kwargs):
        super().__init__(join(root, 'wf-spring'), *args, 
                length=9000, defen_multiples=20, include_unm=True, **kwargs)


class WFSubpages24(Subpages):
    def __init__(self, root, *args, **kwargs):
        super().__init__(join(root, 'subpages24x90x90'), *args, 
                length=5000, **kwargs)
