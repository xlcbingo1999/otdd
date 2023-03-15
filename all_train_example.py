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
import random

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
    parser.add_argument("--mix_two_ratio", type=float, default=0.0)

    args = parser.parse_args()
    return args

args = get_df_config()
raw_data_path = '/mnt/linuxidc_client/dataset/Amazon_Review_split/EMNIST'
sub_train_config_path = '/mnt/linuxidc_client/dataset/Amazon_Review_split/EMNIST_sub_train_datasets_config_12_10.0.json'
result_prefix = '/mnt/linuxidc_client/dataset/Amazon_Review_split/EMNIST/test_otdd_results_all_tr'
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

with open(sub_train_config_path, 'r+') as f:
    current_subtrain_config = json.load(f)
    f.close()
# all_train_ids = sorted(list(current_subtrain_config[dataset_name].keys()))
# all_test_ids = sorted(list(current_subtest_config[dataset_name].keys()))
train_id = args.train_id
test_id = args.test_id
mix_two_ratio = args.mix_two_ratio
device = 'cuda:{}'.format(args.device_index)

sub_train_key = 'train_sub_{}'.format(train_id)
sub_test_key = 'train_sub_{}'.format(test_id)
result_file_name = result_prefix + '/train_{}_test_{}_mix_two_ratio_{}'.format(train_id, test_id, mix_two_ratio)
left_origin_list = list(current_subtrain_config[dataset_name][sub_train_key]["indexes"])
right_origin_list = list(current_subtrain_config[dataset_name][sub_test_key]["indexes"])

if mix_two_ratio > 0.0:
    sample_from_left_indexes = random.sample(left_origin_list, int(len(left_origin_list) * mix_two_ratio))
    sample_from_right_indexes = random.sample(right_origin_list, int(len(right_origin_list) * (1 - mix_two_ratio)))
else:
    sample_from_left_indexes = []
    sample_from_right_indexes = right_origin_list

real_left_origin_list = left_origin_list
real_right_origin_list = list(set.union(set(sample_from_left_indexes), set(sample_from_right_indexes)))

real_train_index = sorted(real_left_origin_list)
real_test_index = sorted(real_right_origin_list)
with open(result_file_name, 'w+') as f:
    print("check last real_train_index: {}".format(real_train_index[-1]), file=f)
    print(len(real_train_index), file=f)
    print("check last real_test_index: ".format(real_test_index[-1]), file=f)
    print(len(real_test_index), file=f)

    print("begin train: {} test: {}".format(train_id, test_id), file=f)
    print("check all size: train[{}]".format(len(train_dataset), file=f))

train_dataset = CustomDataset(train_dataset, real_train_index)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
test_dataset = CustomDataset(train_dataset, real_test_index)
test_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
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
