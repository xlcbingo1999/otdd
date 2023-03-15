import torch
from torchvision.models import resnet18
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torchvision import models


from otdd.pytorch.datasets import load_torchvision_data_from_indexes
from otdd.pytorch.distance import DatasetDistance, FeatureCost
import numpy as np
import json

import argparse
import time


from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from utils.opacus_engine_tools import get_privacy_dataloader


import json
import time
from operator import add
from functools import reduce


# Load MNIST/CIFAR in 3channels (needed by torchvision models)
# loaders_src = load_torchvision_data('CIFAR10', resize=28, maxsize=2000)[0]
# loaders_tgt = load_torchvision_data('MNIST', resize=28, to3channels=True, maxsize=2000)[0]

class CNN(nn.Module):
    def __init__(self, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=3,  # 输入通道数
                out_channels=16,  # 输出通道数
                kernel_size=5,   # 卷积核大小
                stride=1,  #卷积步数
                padding=2,  # 如果想要 con2d 出来的图片长宽没有变化, 
                            # padding=(kernel_size-1)/2 当 stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, output_dim)  # 全连接层，A/Z,a/z一共37个类

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output

def accuracy(preds, labels):
    return (preds == labels).mean()

def get_df_config():
    parser = argparse.ArgumentParser(
                description="Sweep through lambda values")
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--train_dataset_name", type=str, default="EMNIST")
    parser.add_argument("--test_dataset_names", type=str, nargs="+", default=["USPS", "SVHN"]) # SVHN, USPS, STL10, CIFAR10, KMNIST, FashionMNIST
    parser.add_argument("--test_sample_nums", type=int, nargs="+", default=[1500, 500])
    parser.add_argument("--emnist_train_id", type=int, default=0)
    parser.add_argument("--emnist_test_id", type=int, default=1)
    parser.add_argument("--mix_two_ratio", type=float, default=0.0)
    parser.add_argument("--EPSILON", type=float, default=5.0)
    parser.add_argument("--model_name", type=str, default="CNN") # resnet

    args = parser.parse_args()
    return args

args = get_df_config()

train_dataset_name = args.train_dataset_name
emnist_train_id = args.emnist_train_id
# test_dataset_name = args.test_dataset_name
test_dataset_names = args.test_dataset_names
emnist_test_id = args.emnist_test_id
test_sample_nums = args.test_sample_nums

train_sample_num = 18000
test_sample_num = 2000
distance_batch_size = 1024
calculate_batch_size = 1024

DEVICE_INDEX = args.device_index
device = 'cuda:{}'.format(DEVICE_INDEX)
MODEL_NAME = args.model_name
EPOCHS = 50
DEVICE_INDEX = args.device_index
LR = 1e-3
EPSILON = args.EPSILON
DELTA = 1e-7
MAX_GRAD_NORM = 1.2
if MODEL_NAME == "CNN":
    BATCH_SIZE = 1024
    MAX_PHYSICAL_BATCH_SIZE = int(BATCH_SIZE / 2)
else:
    BATCH_SIZE = 64
    MAX_PHYSICAL_BATCH_SIZE = 64


current_time =  time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
if train_dataset_name == 'EMNIST' or 'EMNIST' in test_dataset_names:
    result_file_name = '/mnt/linuxidc_client/otdd/{}_{}_{}_{}_{}_{}_{}.log'.format(MODEL_NAME, EPSILON, train_dataset_name, emnist_train_id, '-'.join(test_dataset_names), '-'.join(map(str, test_sample_nums)), current_time)
    summary_writer_path = '/mnt/linuxidc_client/tensorboard_20230314_otdd/{}_{}_{}_{}_{}_{}_{}'.format(MODEL_NAME, EPSILON, train_dataset_name, emnist_train_id,  '-'.join(test_dataset_names), '-'.join(map(str, test_sample_nums)), current_time)
else:
    result_file_name = '/mnt/linuxidc_client/otdd/{}_{}_{}_{}_{}_{}.log'.format(MODEL_NAME, EPSILON, train_dataset_name, '-'.join(test_dataset_names), '-'.join(map(str, test_sample_nums)), current_time)
    summary_writer_path = '/mnt/linuxidc_client/tensorboard_20230314_otdd/{}_{}_{}_{}_{}_{}'.format(MODEL_NAME, EPSILON, train_dataset_name, '-'.join(test_dataset_names), '-'.join(map(str, test_sample_nums)), current_time)


train_dir = None
test_dir = None



if train_dataset_name == 'EMNIST':
    sub_train_key = 'train_sub_{}'.format(emnist_train_id)
    train_dir = '/mnt/linuxidc_client/dataset'
    sub_train_config_path = '/mnt/linuxidc_client/dataset/sub_train_datasets_config.json'
    with open(sub_train_config_path, 'r+') as f:
        current_subtrain_config = json.load(f)
        f.close()
    sub_train_origin_indexes_list = list(current_subtrain_config[train_dataset_name][sub_train_key]["indexes"])
    sub_train_origin_indexes_list = np.random.choice(sub_train_origin_indexes_list, train_sample_num, replace=False)

    loaders_src, ratio_src, train_dataset = load_torchvision_data_from_indexes(train_dataset_name, target_indexes=sub_train_origin_indexes_list, sample_num=None, target_type='train', batch_size=distance_batch_size, resize=28, to3channels=True, datadir=train_dir)
else:
    loaders_src, ratio_src, train_dataset = load_torchvision_data_from_indexes(train_dataset_name, target_indexes=None, sample_num=train_sample_num, target_type='train', batch_size=distance_batch_size, resize=28, to3channels=True, datadir=train_dir)

all_test_datasets = []
for i, name in enumerate(test_dataset_names):
    if name == 'EMNIST':
        sub_test_key = 'test_sub_{}'.format(emnist_test_id)
        test_dir = '/mnt/linuxidc_client/dataset'
        sub_test_config_path = '/mnt/linuxidc_client/EMNIST/EMNIST_sub_train_datasets_config_4_10.0.json'
        
        with open(sub_test_config_path, 'r+') as f:
            current_subtest_config = json.load(f)
            f.close()

        sub_test_origin_indexes_list = list(current_subtest_config[name][sub_test_key]["indexes"])
        sub_test_origin_indexes_list = np.random.choice(sub_test_origin_indexes_list, test_sample_nums[i], replace=False)

        _, _, test_dataset = load_torchvision_data_from_indexes(name, target_indexes=sub_test_origin_indexes_list, sample_num=None, target_type='test', batch_size=distance_batch_size, resize=28, to3channels=True, datadir=test_dir)
    else:
        _, _, test_dataset = load_torchvision_data_from_indexes(name, target_indexes=None, sample_num=test_sample_nums[i], target_type='test', batch_size=distance_batch_size, resize=28, to3channels=True, datadir=test_dir)
    all_test_datasets.append(test_dataset)
test_dataset = ConcatDataset(all_test_datasets)
loaders_tgt = DataLoader(test_dataset)

embedder = resnet18(pretrained=True).eval()
embedder.fc = torch.nn.Identity()
for p in embedder.parameters():
    p.requires_grad = False

# Here we use same embedder for both datasets
feature_cost = FeatureCost(src_embedding = embedder,
                        src_dim = (3,28,28),
                        tgt_embedding = embedder,
                        tgt_dim = (3,28,28),
                        p = 2,
                        device=device)

dist = DatasetDistance(loaders_src, loaders_tgt,
                        inner_ot_method = 'exact',
                        debiased_loss = True,
                        feature_cost = feature_cost,
                        sqrt_method = 'spectral',
                        sqrt_niters=10,
                        precision='single',
                        p = 2, entreg = 1e-1,
                        device=device,
                        batch_size=calculate_batch_size)

begin = time.time()
d = dist.distance(maxsamples = 10000)
end = time.time()

with open(result_file_name, 'a+') as f:
    print("Total time: {} s".format(end - begin))
    print(f'Embedded OTDD({train_dataset_name},{test_dataset_names})={d:8.2f}')
    print(f'test_sample_nums: {test_sample_nums}')
    print("Total time: {} s".format(end - begin), file=f)
    print(f'Embedded OTDD({train_dataset_name},{test_dataset_names})={d:8.2f}', file=f)
    print(f'test_sample_nums: {test_sample_nums}', file=f)
del embedder

device = torch.device("cuda:{}".format(DEVICE_INDEX) if torch.cuda.is_available() else "cpu")

if MODEL_NAME == "CNN":
    model = CNN(output_dim=len(train_dataset.classes))
elif MODEL_NAME == "resnet":
    model = models.resnet18(num_classes=len(train_dataset.classes))

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # optimize all cnn parameters


privacy_engine = PrivacyEngine() if EPSILON > 0.0 else None
model, optimizer, train_loader = \
    get_privacy_dataloader(privacy_engine, model, optimizer, 
                            loaders_src, EPOCHS, 
                            EPSILON, DELTA, MAX_GRAD_NORM) 

summary_writer = SummaryWriter(summary_writer_path)
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = []
    total_train_acc = []
    temp_debug_tensor = torch.zeros(size=(len(train_dataset.classes), ))
    if privacy_engine is not None:
        with BatchMemoryManager(
            data_loader=train_loader, 
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
            optimizer=optimizer
        ) as memory_safe_data_loader:
            for i, (inputs, labels) in enumerate(memory_safe_data_loader):
                # temp_dis = labels.unique(return_counts=True)
                # temp_key = temp_dis[0]
                # temp_value = temp_dis[1]
                # for index in range(len(temp_key)):
                #     temp_debug_tensor[temp_key[index]] += temp_value[index]
                inputs = inputs.to(device)
                labels = labels.to(device)
                output = model(inputs)
                loss = criterion(output, labels)
                total_train_loss.append(loss.item())

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = labels.detach().cpu().numpy()
                acc = accuracy(preds, labels)
                total_train_acc.append(acc)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % 10 == 0:
                    with open(result_file_name, 'a+') as f:
                        print("epoch[{}]: temp_train_loss: {}".format(epoch, np.mean(total_train_loss)))
                        print("epoch[{}]: temp_train_acc: {}".format(epoch, np.mean(total_train_acc)))
                        print("epoch[{}]: temp_train_loss: {}".format(epoch, np.mean(total_train_loss)), file=f)
                        print("epoch[{}]: temp_train_acc: {}".format(epoch, np.mean(total_train_acc)), file=f)
                    
    else:
        for i, (inputs, labels) in enumerate(train_loader):
            # print("check inputs: {}, labels: {}".format(inputs, labels))
            # temp_dis = labels.unique(return_counts=True)
            # temp_key = temp_dis[0]
            # temp_value = temp_dis[1]
            # for index in range(len(temp_key)):
            #     temp_debug_tensor[temp_key[index]] += temp_value[index]
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)
            loss = criterion(output, labels)
            total_train_loss.append(loss.item())

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = labels.detach().cpu().numpy()
            acc = accuracy(preds, labels)
            total_train_acc.append(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                with open(result_file_name, 'a+') as f:
                    print("epoch[{}]: temp_train_loss: {}".format(epoch, np.mean(total_train_loss)))
                    print("epoch[{}]: temp_train_acc: {}".format(epoch, np.mean(total_train_acc)))
                    print("epoch[{}]: temp_train_loss: {}".format(epoch, np.mean(total_train_loss)), file=f)
                    print("epoch[{}]: temp_train_acc: {}".format(epoch, np.mean(total_train_acc)), file=f)
    
    if privacy_engine is not None:
        epsilon = privacy_engine.get_epsilon(DELTA)
    else:
        epsilon = 0.0
    with open(result_file_name, 'a+') as f:
        print("epoch[{}]: total_train_loss: {}".format(epoch, np.mean(total_train_loss)))
        print("epoch[{}]: total_train_acc: {}".format(epoch, np.mean(total_train_acc)))
        print("epoch[{}]: epsilon_consume: {}".format(epoch, epsilon))
        print("epoch[{}]: total_train_loss: {}".format(epoch, np.mean(total_train_loss)), file=f)
        print("epoch[{}]: total_train_acc: {}".format(epoch, np.mean(total_train_acc)), file=f)
        print("epoch[{}]: epsilon_consume: {}".format(epoch, epsilon), file=f)
    summary_writer.add_scalar('total_train_loss', np.mean(total_train_loss), epoch)
    summary_writer.add_scalar('total_train_acc', np.mean(total_train_acc), epoch)
    summary_writer.add_scalar('epsilon_consume', epsilon, epoch)

    model.eval()
    total_val_loss = []
    total_val_acc = []
    for i, (inputs, labels) in enumerate(loaders_tgt):
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs)
        loss = criterion(output, labels)
        total_val_loss.append(loss.item())

        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = labels.detach().cpu().numpy()
        acc = accuracy(preds, labels)
        total_val_acc.append(acc)
        if (i + 1) % 1000 == 0:
            with open(result_file_name, 'a+') as f:
                print("val epoch[{}]: temp_val_loss: {}".format(epoch, np.mean(total_val_loss)))
                print("val epoch[{}]: temp_val_acc: {}".format(epoch, np.mean(total_val_acc)))
                print("val epoch[{}]: temp_val_loss: {}".format(epoch, np.mean(total_val_loss)), file=f)
                print("val epoch[{}]: temp_val_acc: {}".format(epoch, np.mean(total_val_acc)), file=f)
    with open(result_file_name, 'a+') as f:
        print("val epoch[{}]: total_val_loss: {}".format(epoch, np.mean(total_val_loss)))
        print("val epoch[{}]: total_val_acc: {}".format(epoch, np.mean(total_val_acc)))
        print("val epoch[{}]: total_val_loss: {}".format(epoch, np.mean(total_val_loss)), file=f)
        print("val epoch[{}]: total_val_acc: {}".format(epoch, np.mean(total_val_acc)), file=f)
    summary_writer.add_scalar('total_val_loss', np.mean(total_val_loss), epoch)
    summary_writer.add_scalar('total_val_acc', np.mean(total_val_acc), epoch)

time.sleep(5)