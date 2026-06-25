from scipy import io as scio
from torch.utils.data import DataLoader, Dataset
from opt import *
import os
from torch.utils.data import WeightedRandomSampler

os.environ['TORCH_HOME'] = './'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class eegDataset(Dataset):
    def __init__(self, data, labels, domain_labels=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels.squeeze(), dtype=torch.long)
        if domain_labels is None:
            self.domain_labels = torch.zeros(len(self.labels), dtype=torch.long)
        else:
            self.domain_labels = torch.tensor(domain_labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index], self.domain_labels[index]


def create_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))

def load_DE_SEED(load_path):
    datasets = scio.loadmat(load_path)
    DE = datasets['DE']  # shape: [feature_dim, samples, bands]
    dataAll = np.transpose(DE, [1, 0, 2])  # shape: [samples, channels, bands]
    labelAll = datasets['labelAll'].flatten() + 1  # labels: 0/1/2 -> 1/2/3
    return dataAll, labelAll

def split_subject_data(data_all, label_all, target_subject_id):
    data_target = data_all[target_subject_id]
    label_target = label_all[target_subject_id]

    source_ids = [i for i in range(data_all.shape[0]) if i != target_subject_id]
    data_source = np.concatenate([data_all[i] for i in source_ids], axis=0)
    label_source = np.concatenate([label_all[i] for i in source_ids], axis=0)

    return data_source, label_source, data_target, label_target


def load_dataloader(data_train, label_train, data_target, label_target, batch_size=64, mixed_domain=False):
    if mixed_domain:
        domain_train = np.zeros(len(label_train), dtype=np.int64)
        train_source_dataset = eegDataset(data_train, label_train, domain_labels=domain_train)

        domain_target = np.ones(len(label_target), dtype=np.int64)
        train_target_dataset = eegDataset(data_target, label_target, domain_labels=domain_target)

        combined_train_dataset = torch.utils.data.ConcatDataset([train_source_dataset, train_target_dataset])

        # 创建平衡采样器
        combined_labels = np.concatenate([label_train, label_target])
        sampler = create_balanced_sampler(combined_labels)

        train_loader = DataLoader(combined_train_dataset, batch_size=batch_size,
                                  sampler=sampler, num_workers=2)
    else:
        domain_train = np.zeros(len(label_train), dtype=np.int64)
        sampler = create_balanced_sampler(label_train)
        train_loader = DataLoader(eegDataset(data_train, label_train, domain_labels=domain_train),
                                  batch_size=batch_size, sampler=sampler, num_workers=2)

    domain_test = np.ones(len(label_target), dtype=np.int64)
    test_dataset = eegDataset(data_target, label_target, domain_labels=domain_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader