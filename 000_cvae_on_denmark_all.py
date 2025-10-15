import os
import pathlib
import random
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import utils

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, rdMolDescriptors, Descriptors
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.EState import Fingerprinter
from rdkit.ML.Descriptors import MoleculeDescriptors

cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0, 'blue'), (1, 'red')])
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0, '#144C87'), (0.5, 'white'), (1, 'red')])

def fp2string(fp, output, fp_type="Others"):

    if fp_type in ["Estate", "EstateIndices"]:
        fp = fp
    elif output == "bit":
        fp = list(fp.GetOnBits())

    elif output == "vect":
        fp = list(fp.ToBitString())
        fp = [1 if val in ["1", 1] else 0 for val in fp]

    elif output == "bool":
        fp = list(fp.ToBitString())
        fp = [1 if val == "1" else -1 for val in fp]

    return fp

# 函数根据RDKit库生成RDKit指纹
def rdkit_fp_calc(smiles, is_sdf=False, fp_type='Avalon', radius=2, fp_length=1024, output='vect'):
    """
    avaliable fp_types:
    -------------------
    ['Avalon','AtomPaires','TopologicalTorsions','MACCSKeys','RDKit','RDKitLinear','LayeredFingerprint','Morgan','FeaturedMorgan',"Estate","EstateIndices"]

    Return:
    -------
    type: np.array
    """

    fps = []
    for smi in smiles:
        fp = rdkit_fingerprint(smi, fp_type=fp_type, radius=radius, fp_length=fp_length, output=output)
        fps.append(fp)

    return torch.tensor(np.array(fps))

# 使用RDKit库生成给定分子的指纹（fingerprint）。它接受不同的指纹类型，如Avalon、AtomPaires、TopologicalTorsions、MACCSKeys等。根据提供的指纹类型，它使用RDKit库中相应的方法生成指纹
def rdkit_fingerprint(smi, fp_type="rdkit", radius=2, max_path=2, fp_length=1024, output="vect"):
    """ Molecular fingerprint generation by rdkit package.

    Parameters:
    ------------
    smi: str
        SMILES string.
    fp_type: str
        • Avalon -- Avalon Fingerprint
        • AtomPaires -- Atom-Pairs Fingerprint
        • TopologicalTorsions -- Topological-Torsions Fingerprint
        • MACCSKeys Fingerprint 167
        • RDKit -- RDKit Fingerprint
        • RDKitLinear -- RDKit linear Fingerprint
        • LayeredFingerprint -- RDKit layered Fingerprint
        • Morgan -- Morgan-Circular Fingerprint
        • FeaturedMorgan -- Morgan-Circular Fingerprint with feature definitions
    radius: int
    max_path: int
    fp_length: int
    output: str
        "bit" -- the index of fp exist
        "vect" -- represeant by 0,1
        "bool" -- represeant by 1,-1

    Returns:
    -------
    type: list
        If SMILES is valid, a list will be returned.
        If SMILES is not valid, a list containing zero or Flase will be returned.

    Source:
    -------
    RDKit: https://www.rdkit.org/
    """

    mol = Chem.MolFromSmiles(smi)

    if mol:
        if fp_type == "RDKit":
            fp = Chem.RDKFingerprint(
                mol=mol, maxPath=max_path, fpSize=fp_length)

        elif fp_type == "RDKitLinear":
            fp = Chem.RDKFingerprint(
                mol=mol, maxPath=max_path, branchedPaths=False, fpSize=fp_length)

        elif fp_type == "AtomPaires":
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
                mol, nBits=fp_length)

        elif fp_type == "TopologicalTorsions":
            fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
                mol, nBits=fp_length)

        elif fp_type == "MACCSKeys":
            fp = MACCSkeys.GenMACCSKeys(mol)

        elif fp_type == "Morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius, nBits=fp_length)

        elif fp_type == "FeaturedMorgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius, useFeatures=True, nBits=fp_length)

        elif fp_type == "Avalon":
            fp = pyAvalonTools.GetAvalonFP(mol, nBits=fp_length)

        elif fp_type == "LayeredFingerprint":
            fp = Chem.LayeredFingerprint(
                mol, maxPath=max_path, fpSize=fp_length)

        elif fp_type == "Estate":
            fp = list(Fingerprinter.FingerprintMol(mol)[0])

        elif fp_type == "EstateIndices":
            fp = list(Fingerprinter.FingerprintMol(mol)[1])

        else:
            print("Invalid fingerprint type!")

        fp = fp2string(fp, output, fp_type)

    else:
        if fp_type == "MACCSKeys":
            fp_length = 167
        if fp_type == "Estate":
            fp_length = 79
        if fp_type == "EstateIndices":
            fp_length = 79
        fp = ExplicitBitVect(fp_length)
        fp = fp2string(fp, output='vect')

    return fp

class VAEData(data.Dataset):  # 继承Dataset
    def __init__(self, data_list, imine_list, thiol_list, tag_list):
        self.data = data_list
        self.imine_list = imine_list
        self.thiol_list = thiol_list
        self.labels = tag_list
        # set transformer
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):  # 返回整个数据集的大小
        return len(self.data)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image = self.data[index].astype(float)  # 根据索引index获取该图片
        imine = self.imine_list[index].astype(float)  # 根据索引index获取该图片
        thiol = self.thiol_list[index].astype(float)  # 根据索引index获取该图片
        img = self.transform(image)
        imine = self.transform(imine)
        thiol = self.transform(thiol)

        label = self.labels[index].astype(float)  # 获取该图片的label

        sample = {'img': img, 'imine': imine, 'thiol': thiol, 'label': label}  # 根据图片和标签创建字典

        return sample  # 返回该样本

class VAEData_smile(data.Dataset):  # 继承Dataset
    def __init__(self, data_list, imine_list, thiol_list, tag_list):
        self.data = data_list
        self.imine_list = imine_list
        self.thiol_list = thiol_list
        self.labels = tag_list
        # set transformer
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):  # 返回整个数据集的大小
        return len(self.data)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image = self.data[index].astype(float)  # 根据索引index获取该图片
        imine = self.imine_list[index]  # 根据索引index获取该图片
        thiol = self.thiol_list[index]  # 根据索引index获取该图片
        img = self.transform(image)
        # imine = self.transform(imine)
        # thiol = self.transform(thiol)

        label = self.labels[index].astype(float)  # 获取该图片的label

        sample = {'img': img, 'imine': imine, 'thiol': thiol, 'label': label}  # 根据图片和标签创建字典

        return sample  # 返回该样本


class CVAE(nn.Module):
    """Implementation of CVAE(Conditional Variational Auto-Encoder)"""
    def __init__(self, feature_size, class_size, latent_size):
        super(CVAE, self).__init__()

        # self.l1 = nn.Linear(1, 2*class_size)
        # self.l2 = nn.Linear(2*class_size, class_size)

        self.label_embedding = torch.nn.Embedding(50, class_size)

        self.imine_embedding = nn.Sequential(
            nn.Linear(feature_size, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
        )
        self.thiol_embedding = nn.Sequential(
            nn.Linear(feature_size, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
        )

        self.fc1 = nn.Linear(feature_size + 100 + class_size, 200)
        self.fc2_mu = nn.Linear(200, latent_size)
        self.fc2_log_std = nn.Linear(200, latent_size)
        self.fc3 = nn.Linear(latent_size + 100 + class_size, 200)
        self.fc4 = nn.Linear(200, feature_size)

    def encode(self, x, imine_fea, thiol_fea, y):
        # y = y.unsqueeze(-1)
        # y = self.l2(F.relu(self.l1(y)))
        y = self.label_embedding(y.long())
        imine_fea = self.imine_embedding(imine_fea)
        thiol_fea = self.imine_embedding(thiol_fea)
        h1 = F.relu(self.fc1(torch.cat([x, imine_fea, thiol_fea, y], dim=1)))  # concat features and labels
        mu = self.fc2_mu(h1)
        log_std = self.fc2_log_std(h1)
        return mu, log_std

    def decode(self, z, imine_fea, thiol_fea, y):
        # y = y.unsqueeze(-1)
        # y = self.l2(F.relu(self.l1(y)))
        y = self.label_embedding(y.long())
        imine_fea = self.imine_embedding(imine_fea)
        thiol_fea = self.imine_embedding(thiol_fea)
        h3 = F.relu(self.fc3(torch.cat([z, imine_fea, thiol_fea, y], dim=1)))  # concat latents and labels
        recon = torch.sigmoid(self.fc4(h3))  # use sigmoid because the input image's pixel is between 0-1
        return recon

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # simple from standard normal distribution
        z = mu + eps * std
        return z

    def forward(self, x, imine_fea, thiol_fea, y):
        mu, log_std = self.encode(x, imine_fea, thiol_fea, y)
        z = self.reparametrize(mu, log_std)
        recon = self.decode(z, imine_fea, thiol_fea, y)
        return recon, mu, log_std

    def loss_function(self, recon, x, mu, log_std) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, x, reduction="sum")  # use "mean" may have a bad effect on gradients
        kl_loss = -0.5 * (1 + 2*log_std - mu.pow(2) - torch.exp(2*log_std))
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss + kl_loss
        return loss

class CVAE_Pre(nn.Module):
    """Implementation of CVAE(Conditional Variational Auto-Encoder)"""
    def __init__(self, feature_size, class_size, latent_size):
        super(CVAE_Pre, self).__init__()

        # self.l1 = nn.Linear(1, 2*class_size)
        # self.l2 = nn.Linear(2*class_size, class_size)

        self.label_embedding = torch.nn.Embedding(50 + 2, class_size)

        self.imine_embedding = nn.Sequential(
            nn.Linear(feature_size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
        )
        self.thiol_embedding = nn.Sequential(
            nn.Linear(feature_size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
        )

        self.fc1 = nn.Linear(feature_size + 200 + class_size, 200)
        self.fc2_mu = nn.Linear(200, latent_size)
        self.fc2_log_std = nn.Linear(200, latent_size)
        self.fc3 = nn.Linear(latent_size + 200 + class_size, 200)
        self.fc4 = nn.Linear(200, feature_size)

        self.predictor = nn.Sequential(
            nn.Linear(feature_size + 200, 500),
            # nn.Tanh(), # the authors of the paper used tanh
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(500, 500),
            # nn.Tanh(), # the authors of the paper used tanh
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(500, 1)
        )

    def encode(self, x, imine_fea, thiol_fea, y):
        # y = y.unsqueeze(-1)
        # y = self.l2(F.relu(self.l1(y)))
        y = self.label_embedding(y.long())
        imine_fea = self.imine_embedding(imine_fea)
        thiol_fea = self.imine_embedding(thiol_fea)
        h1 = F.relu(self.fc1(torch.cat([x, imine_fea, thiol_fea, y], dim=1)))  # concat features and labels
        mu = self.fc2_mu(h1)
        log_std = self.fc2_log_std(h1)
        return mu, log_std

    def decode(self, z, imine_fea, thiol_fea, y):
        # y = y.unsqueeze(-1)
        # y = self.l2(F.relu(self.l1(y)))
        y = self.label_embedding(y.long())
        imine_fea = self.imine_embedding(imine_fea)
        thiol_fea = self.imine_embedding(thiol_fea)
        h3 = F.relu(self.fc3(torch.cat([z, imine_fea, thiol_fea, y], dim=1)))  # concat latents and labels
        recon = torch.sigmoid(self.fc4(h3))  # use sigmoid because the input image's pixel is between 0-1
        return recon

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # simple from standard normal distribution
        z = mu + eps * std
        return z

    def forward(self, x, imine_fea, thiol_fea, y):
        mu, log_std = self.encode(x, imine_fea, thiol_fea, y)
        z = self.reparametrize(mu, log_std)
        recon = self.decode(z, imine_fea, thiol_fea, y)
        return recon, mu, log_std

    def predict(self, recon, imine_fea, thiol_fea):
        imine_fea = self.imine_embedding(imine_fea)
        thiol_fea = self.imine_embedding(thiol_fea)
        return self.predictor(torch.cat([recon, imine_fea, thiol_fea], dim=1))

    def kl_anneal_function(self, epoch, anneal_start=29, k=1):
        return 1 / (1 + np.exp(- k * (epoch - anneal_start)))

    def loss_function(self, recon, x, mu, log_std, logits, y, epoch):
        reconstruction_loss_func = nn.NLLLoss()
        reg_loss_func = nn.MSELoss()

        recon_loss = F.mse_loss(recon, x, reduction="sum")  # use "mean" may have a bad effect on gradients
        # recon_loss = reconstruction_loss_func(recon, x)

        kl_loss = -0.5 * (1 + 2*log_std - mu.pow(2) - torch.exp(2*log_std))
        kl_loss = torch.sum(kl_loss)
        kl_weight = self.kl_anneal_function(epoch)

        reg_loss = reg_loss_func(logits.flatten(), y)
        reg_loss = reg_loss.type(torch.float32)

        loss = recon_loss + kl_loss + reg_loss
        # loss = recon_loss + kl_loss * kl_weight + reg_loss

        return loss, recon_loss, kl_loss, reg_loss

class CVAE_smile(nn.Module):
    """Implementation of CVAE(Conditional Variational Auto-Encoder)"""
    def __init__(self, feature_size, class_size, latent_size):
        super(CVAE_smile, self).__init__()

        # self.l1 = nn.Linear(1, 2*class_size)
        # self.l2 = nn.Linear(2*class_size, class_size)

        self.label_embedding = torch.nn.Embedding(50 + 2, class_size)

        # self.imine_embedding = nn.Sequential(
        #     nn.Linear(feature_size, 100),
        #     nn.ReLU(),
        # )
        # self.thiol_embedding = nn.Sequential(
        #     nn.Linear(feature_size, 100),
        #     nn.ReLU(),
        # )

        self.fc1 = nn.Linear(feature_size + 32*2 + class_size, 200)
        self.fc2_mu = nn.Linear(
            200, latent_size)
        self.fc2_log_std = nn.Linear(200, latent_size)
        self.fc3 = nn.Linear(latent_size + 32*2 + class_size, 200)
        self.fc4 = nn.Linear(200, feature_size)

    def encode(self, x, imine_fea, thiol_fea, y):
        # y = y.unsqueeze(-1)
        # y = self.l2(F.relu(self.l1(y)))
        y = self.label_embedding(y)
        # y = self.label_embedding(y.long())
        # imine_fea = self.imine_embedding(imine_fea)
        # thiol_fea = self.imine_embedding(thiol_fea)
        h1 = F.relu(self.fc1(torch.cat([x, imine_fea, thiol_fea, y], dim=1)))  # concat features and labels
        mu = self.fc2_mu(h1)
        log_std = self.fc2_log_std(h1)
        return mu, log_std

    def decode(self, z, imine_fea, thiol_fea, y):
        # y = y.unsqueeze(-1)
        # y = self.l2(F.relu(self.l1(y)))
        y = self.label_embedding(y.long())
        # imine_fea = self.imine_embedding(imine_fea)
        # thiol_fea = self.imine_embedding(thiol_fea)
        h3 = F.relu(self.fc3(torch.cat([z, imine_fea, thiol_fea, y], dim=1)))  # concat latents and labels
        recon = torch.sigmoid(self.fc4(h3))  # use sigmoid because the input image's pixel is between 0-1
        return recon

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # simple from standard normal distribution
        z = mu + eps * std
        return z

    def forward(self, x, imine_fea, thiol_fea, y):
        mu, log_std = self.encode(x, imine_fea, thiol_fea, y)
        z = self.reparametrize(mu, log_std)
        recon = self.decode(z, imine_fea, thiol_fea, y)
        return recon, mu, log_std

    def loss_function(self, recon, x, mu, log_std) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, x, reduction="sum")  # use "mean" may have a bad effect on gradients
        kl_loss = -0.5 * (1 + 2*log_std - mu.pow(2) - torch.exp(2*log_std))
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss + kl_loss
        return loss

def generate_edges(num_categories, min = -1, max = 3):
    # 确定每个类别的宽度
    width = (max - min) / num_categories

    # 生成边界数组
    edges = [i * width + min for i in range(num_categories)]

    # 添加最后一个边界值
    edges.append(max)  # 假设最大值为3

    return edges

def reverse_map(categories, edges):
    # 创建一个与类别数组相同大小的数组，用于存储反映射后的浮点数值
    reversed_labels = np.zeros_like(categories, dtype=float)

    # 遍历类别数组
    for i, category in enumerate(categories):
        # 使用类别值作为索引，从边界数组中获取对应的值
        reversed_labels[i] = (edges[category - 1] + edges[category]) / 2.0 if category > 0 else edges[0]

    return reversed_labels

def train(epochs, batch_size, lr, train_data, root_path, device):

    recon = None
    img = None

    utils.make_dir(f"{root_path}/img/cvae")
    utils.make_dir(f"{root_path}/model_weights/cvae")

    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    cvae = CVAE(feature_size=800, class_size=10, latent_size=10).to(device)

    optimizer = torch.optim.Adam(cvae.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss = 0
        i = 0
        for batch_id, data in enumerate(data_loader):
            img = data['img'].float().to(device)
            imine = data['imine'].float().to(device)
            thiol = data['thiol'].float().to(device)
            label = data['label'].float()  # 还需要加一下.float()
            inputs = img.reshape(img.shape[0], -1)
            imine_fea = imine.reshape(imine.shape[0], -1)
            thiol_fea = thiol.reshape(thiol.shape[0], -1)
            y = label.to(device)
            recon, mu, log_std = cvae(inputs, imine_fea, thiol_fea, y)
            loss = cvae.loss_function(recon, inputs, mu, log_std)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            i += 1

            if batch_id % 1 == 0:
                print("Epoch[{}/{}], Batch[{}/{}], batch_loss:{:.6f}".format(
                    epoch + 1, epochs, batch_id + 1, len(data_loader), loss.item()))

        print("======>epoch:{},\t epoch_average_batch_loss:{:.6f}============".format(epoch + 1, train_loss / i), "\n")

        # save imgs
        if epoch % 10 == 0:
            imgs = recon.detach().cpu().numpy()
            path = f"{root_path}/img/cvae/epoch{epoch + 1}.png"
            # torchvision.utils.save_image(imgs, path, nrow=10)
            utils.to_heatmap(imgs[:9], path)
            print("save:", path, "\n")

    # torchvision.utils.save_image(img, f"{root_path}/img/cvae/raw.png", nrow=10)
    utils.to_heatmap(img.cpu().numpy(), f"{root_path}/img/cvae/raw.png")
    print("save raw image:./img/cvae/raw/png", "\n")

    # save val model
    utils.save_model(cvae, f"{root_path}/model_weights/cvae/cvae_weights.pth")

def train_predict(epochs, batch_size, lr, train_data, root_path, device):

    MODELS_DIR = './losses'
    models_dir = pathlib.Path(MODELS_DIR)
    models_dir.mkdir(exist_ok=True, parents=True)
    # self.encoder_file = models_dir / pathlib.Path(f'{reg_col}_encoder.pth')
    # self.decoder_file = models_dir / pathlib.Path(f'{reg_col}_decoder.pth')
    property_predictor_file = models_dir / pathlib.Path(f'property_predictor.pth')
    train_losses_file = models_dir / pathlib.Path(f'train_losses.csv')
    # self.val_losses_file = models_dir / pathlib.Path(f'{reg_col}_valid_losses.csv')

    loss_columns = ['total_loss', 'reconstruction_loss', 'kl_divergence', 'regression_loss']
    train_losses_df = pd.DataFrame(columns=loss_columns)
    # self.val_losses_df = pd.DataFrame(columns=self.loss_columns)

    recon = None
    img = None

    utils.make_dir(f"{root_path}/img/cvae")
    utils.make_dir(f"{root_path}/model_weights/cvae")

    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    cvae = CVAE_Pre(feature_size=800, class_size=50, latent_size=100).to(device)

    # 定义类别的边界
    edges = generate_edges(50, -1, 3)  # 50个类别

    optimizer = torch.optim.Adam(cvae.parameters(), lr=lr)
    train_losses = []
    for epoch in range(epochs):
        train_loss = 0
        recon_loss_epoch = 0
        kld_epoch = 0
        reg_loss_epoch = 0
        i = 0
        num_loaders = len(data_loader)
        for batch_id, data in enumerate(data_loader):
            img = data['img'].float().to(device)
            imine = data['imine'].float().to(device)
            thiol = data['thiol'].float().to(device)
            label_float = data['label'].float().to(device)  # 还需要加一下.float()
            # 使用digitize函数将浮点型数据映射到离散的类别
            label = torch.tensor(np.digitize(data['label'].numpy(), edges)).long()
            inputs = img.reshape(img.shape[0], -1)
            imine_fea = imine.reshape(imine.shape[0], -1)
            thiol_fea = thiol.reshape(thiol.shape[0], -1)
            y = label.to(device)
            recon, mu, log_std = cvae(inputs, imine_fea, thiol_fea, y)
            logits = cvae.predict(recon, imine_fea, thiol_fea)
            loss, recon_loss, kl_loss, reg_loss = cvae.loss_function(recon, inputs, mu, log_std, logits, label_float, epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            recon_loss_epoch += recon_loss.item()
            kld_epoch += kl_loss.item()
            reg_loss_epoch += reg_loss.item()

            i += 1

            if batch_id % 1 == 0:
                print("Epoch[{}/{}], Batch[{}/{}], batch_loss:{:.6f}".format(
                    epoch + 1, epochs, batch_id + 1, len(data_loader), loss.item()))

        train_loss /= num_loaders
        recon_loss_epoch /= num_loaders
        kld_epoch /= num_loaders
        reg_loss_epoch /= num_loaders

        print("======>epoch:{},\t epoch_average_batch_loss:{:.6f}============".format(epoch + 1, train_loss / i), "\n")

        train_losses.append((train_loss, recon_loss_epoch, kld_epoch, reg_loss_epoch))

        # save imgs
        if epoch % 10 == 0:
            imgs = recon.detach().cpu().numpy()
            path = f"{root_path}/img/cvae/epoch{epoch + 1}.png"
            # torchvision.utils.save_image(imgs, path, nrow=10)
            utils.to_heatmap(imgs[:9], path)
            print("save:", path, "\n")

    # torchvision.utils.save_image(img, f"{root_path}/img/cvae/raw.png", nrow=10)
    utils.to_heatmap(img.cpu().numpy(), f"{root_path}/img/cvae/raw.png")
    print("save raw image:./img/cvae/raw/png", "\n")

    # save val model
    utils.save_model(cvae, f"{root_path}/model_weights/cvae/cvae_weights.pth")

    temp_df = pd.DataFrame(data=train_losses, columns=loss_columns)
    train_losses_df = train_losses_df.append(temp_df, ignore_index=True)
    train_losses_df.to_csv(train_losses_file, index=False)

    # plot loss
    train_losses_df = pd.read_csv(train_losses_file)
    plt.rcParams.update({'font.size': 15})
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
    ax1.plot(train_losses_df['total_loss'], label='Train')
    ax1.set_ylabel('Total Loss')
    ax1.legend(loc='upper right')
    ax2.plot(train_losses_df['reconstruction_loss'], label='Train')
    ax2.set_ylabel('Reconstruction Loss')
    ax2.legend(loc='upper right')
    ax3.plot(train_losses_df['kl_divergence'], label='Train')
    ax3.set_yscale('log')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('KL Divergence')
    ax3.legend(loc='upper right')
    ax4.plot(train_losses_df['regression_loss'], label='Train')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Regression Loss')
    ax4.legend(loc='upper right')
    plt.savefig(pathlib.Path(MODELS_DIR, f"losses.png"))
    plt.show()

def train_smile(epochs, batch_size, lr, train_data, root_path, device):

    recon = None
    img = None

    utils.make_dir(f"{root_path}/img/cvae")
    utils.make_dir(f"{root_path}/model_weights/cvae")

    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # cvae = CVAE(feature_size=800, class_size=10, latent_size=10).to(device)
    cvae = CVAE_smile(feature_size=800, class_size=32, latent_size=100).to(device)

    optimizer = torch.optim.Adam(cvae.parameters(), lr=lr)

    # 定义类别的边界
    edges = generate_edges(50, -1, 3) # 50个类别

    for epoch in range(epochs):
        train_loss = 0
        i = 0
        for batch_id, data in enumerate(data_loader):
            img = data['img'].float().to(device)

            # RDKit Fingerprint: 11 types
            # fp_type = ['Avalon', 'AtomPaires', 'TopologicalTorsions', 'MACCSKeys', 'RDKit', 'RDKitLinear', 'LayeredFingerprint', 'Morgan', 'FeaturedMorgan', "Estate", "EstateIndices"]:
            imine = rdkit_fp_calc(smiles=data['imine'], is_sdf=False, fp_type='RDKit', fp_length=32).float().to(device)
            thiol = rdkit_fp_calc(smiles=data['thiol'], is_sdf=False, fp_type='RDKit', fp_length=32).float().to(device)

            # label = data['label'].float()  # 还需要加一下.float()
            # 使用digitize函数将浮点型数据映射到离散的类别
            label = torch.tensor(np.digitize(data['label'].numpy(), edges)).long()

            inputs = img.reshape(img.shape[0], -1)
            imine_fea = imine.reshape(imine.shape[0], -1)
            thiol_fea = thiol.reshape(thiol.shape[0], -1)
            y = label.to(device)
            recon, mu, log_std = cvae(inputs, imine_fea, thiol_fea, y)
            loss = cvae.loss_function(recon, inputs, mu, log_std)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            i += 1

            if batch_id % 1 == 0:
                print("Epoch[{}/{}], Batch[{}/{}], batch_loss:{:.6f}".format(
                    epoch + 1, epochs, batch_id + 1, len(data_loader), loss.item()))

        print("======>epoch:{},\t epoch_average_batch_loss:{:.6f}============".format(epoch + 1, train_loss / i),
              "\n")

        # save imgs
        if epoch % 10 == 0:
            imgs = recon.detach().cpu().numpy()
            path = f"{root_path}/img/cvae/epoch{epoch + 1}.png"
            # torchvision.utils.save_image(imgs, path, nrow=10)
            utils.to_heatmap(imgs[:9], path)
            print("save:", path, "\n")

    # torchvision.utils.save_image(img, f"{root_path}/img/cvae/raw.png", nrow=10)
    utils.to_heatmap(img.cpu().numpy(), f"{root_path}/img/cvae/raw.png")
    print("save raw image:./img/cvae/raw/png", "\n")

    # save val model
    utils.save_model(cvae, f"{root_path}/model_weights/cvae/cvae_weights.pth")

if __name__ == '__main__':

    # 设置随机种子，保证结果可重复
    seed = 222
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 指定运行的GPU显卡，加速运算
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    gpu_ids = [0]

    device = torch.device(f'cuda:{gpu_ids[0]}')

    result_df = pd.read_csv('./Reaction_Result/Denmark_Reaction_Data.csv')

    # 获取每种底物的smiles码
    cat_smiles = result_df['Catalyst'].to_list()
    imine_smiles = result_df['Imine'].to_list()
    thiol_smiles = result_df['Thiol'].to_list()
    # 获取所有反应对应的ddg标签
    ddG = result_df['Output'].to_list()

    # 去除重复的smiles码
    cat_smiles_set = sorted(list(set(cat_smiles)))
    imine_smiles_set = sorted(list(set(imine_smiles)))
    thiol_smiles_set = sorted(list(set(thiol_smiles)))

    # 加载数据
    cat_spms = np.load('Jishe_new_81/20X40/cat.npy')
    imine_spms = np.load('Jishe_new_81/20X40/imine.npy')
    thiol_spms = np.load('Jishe_new_81/20X40/thiol.npy')
    tag = np.load('./Reaction_Result/ddG.npy')

    # 设置Seaborn风格
    sns.set(style="whitegrid")

    # # 绘制年龄分布直方图
    # plt.figure(figsize=(8, 6))
    # sns.histplot(tag, bins=30, kde=True, color='skyblue')
    # plt.title("DDG Distribution")
    # plt.xlabel("DDG")
    # plt.ylabel("Frequency")
    # plt.show()

    epochs = 100
    batch_size = 32
    # root_path = f'/media/data1/Models_ly/3DChemical/denmark-1/generate-100-embbeding/Catalyst_all'
    # root_path = f'/media/data1/Models_ly/3DChemical/denmark-1/generate-100-embbeding/Catalyst_all_smile_4'
    root_path = f'/media/data1/Models_ly/3DChemical/denmark-1/generate-100-embbeding/Catalyst_all_predict'
    train_dataset = VAEData(cat_spms, imine_spms, thiol_spms, tag)
    # train_dataset = VAEData_smile(cat_spms, imine_smiles, thiol_smiles, tag)

    # train(epochs=epochs, batch_size=batch_size, lr=1e-3, train_data=train_dataset, root_path=root_path, device=device)
    # train_smile(epochs=epochs, batch_size=batch_size, lr=1e-3, train_data=train_dataset, root_path=root_path, device=device)
    train_predict(epochs=epochs, batch_size=batch_size, lr=1e-3, train_data=train_dataset, root_path=root_path, device=device)