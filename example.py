# 如何切分
# 每个类最多5000，然后各种切分方案（其中每个subtrain_datablock的数量一致 50000）
# 迪利克雷切分方案 0.1 1.0 10.0 ...
# 相同选择
# train和test是有子集的关系

from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.distance import DatasetDistance, IncomparableDatasetDistance
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Grayscale
import json
import argparse
import time

class CustomDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.targets = dataset.targets # 保留targets属性
        self.classes = dataset.classes # 保留classes属性
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        x, y = self.dataset[self.indices[item]]
        return x, y
    
    def get_class_distribution(self):
        sub_targets = self.targets[self.indices]
        return sub_targets.unique(return_counts=True)

def get_df_config():
    parser = argparse.ArgumentParser(
                description="Sweep through lambda values")
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--train_id", type=int, default=0)
    parser.add_argument("--test_id", type=int, default=0)

    args = parser.parse_args()
    return args

args = get_df_config()
raw_data_path = '/mnt/linuxidc_client/dataset/Amazon_Review_split/EMNIST'
sub_train_config_path = '/mnt/linuxidc_client/dataset/Amazon_Review_split/EMNIST_sub_train_datasets_config_4_1.0.json'
sub_test_config_path = '/mnt/linuxidc_client/dataset/Amazon_Review_split/EMNIST_test_dataset_config_4_10.0.json'
result_prefix = '/mnt/linuxidc_client/dataset/Amazon_Review_split/EMNIST/otdd_results'
dataset_name = "EMNIST"
BATCH_SIZE = 2048

transform = Compose([
    Grayscale(3),
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])
train_dataset = EMNIST(
    root=raw_data_path,
    split="bymerge",
    download=False,
    train=True,
    transform=transform
)
test_dataset = EMNIST(
    root=raw_data_path,
    split="bymerge",
    download=False,
    train=False,
    transform=transform
)

with open(sub_train_config_path, 'r+') as f:
    current_subtrain_config = json.load(f)
    f.close()
with open(sub_test_config_path, 'r+') as f:
    current_subtest_config = json.load(f)
    f.close()
# all_train_ids = sorted(list(current_subtrain_config[dataset_name].keys()))
# all_test_ids = sorted(list(current_subtest_config[dataset_name].keys()))
train_id = args.train_id
test_id = args.test_id
device = 'cuda:{}'.format(args.device_index)

sub_train_key = 'train_sub_{}'.format(train_id)
sub_test_key = 'test_sub_{}'.format(test_id)
result_file_name = result_prefix + '/tr4te1_1.0_10.0_train_{}_test_{}'.format(train_id, test_id)
real_train_index = sorted(list(current_subtrain_config[dataset_name][sub_train_key]["indexes"]))
real_test_index = sorted(list(current_subtest_config[dataset_name][sub_test_key]["indexes"])) 
with open(result_file_name, 'a+') as f:
    print("check last real_train_index: {}".format(real_train_index[-1]), file=f)
    print(len(real_train_index), file=f)
    print("check last real_test_index: ".format(real_test_index[-1]), file=f)
    print(len(real_test_index), file=f)

    print("begin train: {} test: {}".format(train_id, test_id), file=f)
    print("check all size: train[{}] and test[{}]".format(len(train_dataset), len(test_dataset)), file=f)

train_dataset = CustomDataset(train_dataset, real_train_index)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
test_dataset = CustomDataset(test_dataset, real_test_index)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)
with open(result_file_name, 'a+') as f:
    print("Finished split datasets!", file=f)
    print("check train_loader: {}".format(len(train_loader) * BATCH_SIZE), file=f)
    print("check test_loader: {}".format(len(test_loader) * BATCH_SIZE), file=f)


begin = time.time()
# Instantiate distance
dist = DatasetDistance(train_loader, test_loader,
                        inner_ot_method = 'exact',
                        debiased_loss = True,
                        p = 2, entreg = 1e-1,
                        device=device)

d = dist.distance(maxsamples = 10000)
end = time.time()
with open(result_file_name, 'a+') as f:
    print('OTDD-EMNIST(train[{}],test[{}])={}'.format(train_id, test_id, d))
    print('OTDD-EMNIST(train[{}],test[{}])={}'.format(train_id, test_id, d), file=f)
    print("Total time: {} s".format(end - begin))
    print("Total time: {} s".format(end - begin), file=f)
