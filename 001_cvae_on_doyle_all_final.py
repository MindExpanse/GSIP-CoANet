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

cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0, 'blue'), (1, 'red')])
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0, '#144C87'), (0.5, 'white'), (1, 'red')])

class VAEData(data.Dataset):  # 继承Dataset
    def __init__(self, data_list, lig_list, base_list, ar_list, tag_list):
        self.data = data_list
        self.lig_list = lig_list
        self.base_list = base_list
        self.ar_list = ar_list
        self.labels = tag_list
        # set transformer
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):  # 返回整个数据集的大小
        return len(self.data)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image = self.data[index].astype(float)  # 根据索引index获取该图片
        lig = self.lig_list[index].astype(float)  # 根据索引index获取该图片
        base = self.base_list[index].astype(float)  # 根据索引index获取该图片
        ar = self.ar_list[index].astype(float)  # 根据索引index获取该图片
        img = self.transform(image)
        lig = self.transform(lig)
        base = self.transform(base)
        ar = self.transform(ar)

        label = self.labels[index].astype(float)  # 获取该图片的label

        sample = {'img': img, 'lig': lig, 'base': base, 'ar': ar, 'label': label}  # 根据图片和标签创建字典

        return sample  # 返回该样本

class CVAE_Pre(nn.Module):
    """Implementation of CVAE(Conditional Variational Auto-Encoder)"""
    def __init__(self, feature_size, class_size, latent_size):
        super(CVAE_Pre, self).__init__()

        # self.l1 = nn.Linear(1, 2*class_size)
        # self.l2 = nn.Linear(2*class_size, class_size)

        self.label_embedding = torch.nn.Embedding(100 + 2, class_size)

        self.lig_embedding = nn.Sequential(
            nn.Linear(feature_size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
        )
        self.base_embedding = nn.Sequential(
            nn.Linear(feature_size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
        )
        self.ar_embedding = nn.Sequential(
            nn.Linear(feature_size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
        )

        self.fc1 = nn.Linear(feature_size + 300 + class_size, 200)
        self.fc2_mu = nn.Linear(200, latent_size)
        self.fc2_log_std = nn.Linear(200, latent_size)
        self.fc3 = nn.Linear(latent_size + 300 + class_size, 200)
        self.fc4 = nn.Linear(200, feature_size)

        self.predictor = nn.Sequential(
            nn.Linear(feature_size + 300, 500),
            # nn.Tanh(), # the authors of the paper used tanh
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(500, 500),
            # nn.Tanh(), # the authors of the paper used tanh
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(500, 1)
        )

    def encode(self, x, lig_fea, base_fea, ar_fea, y):
        # y = y.unsqueeze(-1)
        # y = self.l2(F.relu(self.l1(y)))
        y = self.label_embedding(y.long())
        lig_fea = self.lig_embedding(lig_fea)
        base_fea = self.base_embedding(base_fea)
        ar_fea = self.ar_embedding(ar_fea)
        h1 = F.relu(self.fc1(torch.cat([x, lig_fea, base_fea, ar_fea, y], dim=1)))  # concat features and labels
        mu = self.fc2_mu(h1)
        log_std = self.fc2_log_std(h1)
        return mu, log_std

    def decode(self, z, lig_fea, base_fea, ar_fea, y):
        # y = y.unsqueeze(-1)
        # y = self.l2(F.relu(self.l1(y)))
        y = self.label_embedding(y.long())
        lig_fea = self.lig_embedding(lig_fea)
        base_fea = self.base_embedding(base_fea)
        ar_fea = self.ar_embedding(ar_fea)
        h3 = F.relu(self.fc3(torch.cat([z, lig_fea, base_fea, ar_fea, y], dim=1)))  # concat latents and labels
        recon = torch.sigmoid(self.fc4(h3))  # use sigmoid because the input image's pixel is between 0-1
        return recon

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # simple from standard normal distribution
        z = mu + eps * std
        return z

    def forward(self, x, lig_fea, base_fea, ar_fea, y):
        y = y.reshape(-1)
        mu, log_std = self.encode(x, lig_fea, base_fea, ar_fea, y)
        z = self.reparametrize(mu, log_std)
        recon = self.decode(z, lig_fea, base_fea, ar_fea, y)
        return recon, mu, log_std

    def predict(self, recon, lig_fea, base_fea, ar_fea):
        lig_fea = self.lig_embedding(lig_fea)
        base_fea = self.base_embedding(base_fea)
        ar_fea = self.ar_embedding(ar_fea)
        return self.predictor(torch.cat([recon, lig_fea, base_fea, ar_fea], dim=1))

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

from torchsummary import summary
# model = CNN()
model = CVAE_Pre(feature_size=800, class_size=50, latent_size=100)
for name, layer in model.named_children():
    print(name, layer)
summary(model, [(800,), (800,), (800,), (800,), (1,)], device='cpu')

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

    cvae = CVAE_Pre(feature_size=800, class_size=100, latent_size=100).to(device)
    # cvae = CVAE_Pre(feature_size=3200, class_size=100, latent_size=100).to(device)

    # 定义类别的边界
    edges = generate_edges(100, -1, 3)  # 100个类别

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
            lig = data['lig'].float().to(device)
            base = data['base'].float().to(device)
            ar = data['ar'].float().to(device)
            label_float = data['label'].float().to(device)  # 还需要加一下.float()
            # 使用digitize函数将浮点型数据映射到离散的类别
            label = torch.tensor(np.digitize(data['label'].numpy(), edges)).long()
            inputs = img.reshape(img.shape[0], -1)
            lig_fea = lig.reshape(lig.shape[0], -1)
            base_fea = base.reshape(base.shape[0], -1)
            ar_fea = ar.reshape(ar.shape[0], -1)
            y = label.to(device)
            recon, mu, log_std = cvae(inputs, lig_fea, base_fea, ar_fea, y)
            logits = cvae.predict(recon, lig_fea, base_fea, ar_fea)
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
            # 将图像扩展为 4D 张量 (N, C, H, W)，这里假设 C = 1（单通道图像）
            large_imgs = recon.reshape(recon.shape[0], 1, 20, 40)

            # 通过双线性插值将图像扩展到 (N, 1, 40, 80)
            large_imgs = F.interpolate(large_imgs, size=(40, 80), mode='bilinear', align_corners=False)

            # 如果需要，还可以将结果 reshape 回 (N, 40, 80)
            large_imgs = large_imgs.squeeze(1)  # (N, 40, 80)

            imgs = recon.detach().cpu().numpy()
            large_imgs = large_imgs.detach().cpu().numpy()

            path = f"{root_path}/img/cvae/epoch{epoch + 1}.png"
            large_path = f"{root_path}/img/cvae/l_epoch{epoch + 1}.png"
            # torchvision.utils.save_image(imgs, path, nrow=10)
            utils.to_heatmap(imgs[:9], path)
            utils.to_heatmap(large_imgs[:9], large_path, True)
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

    # 加载数据
    # cat_spms = np.load('Jishe_new_81/20X40/cat.npy')
    # imine_spms = np.load('Jishe_new_81/20X40/imine.npy')
    # thiol_spms = np.load('Jishe_new_81/20X40/thiol.npy')
    # tag = np.load('./Reaction_Result/ddG.npy')

    lig_spms = np.load('./Doyle/Jishe_8/20X40/lig.npy')
    add_spms = np.load('./Doyle/Jishe_8/20X40/add.npy')
    base_spms = np.load('./Doyle/Jishe_8/20X40/base.npy')
    ar_spms = np.load('./Doyle/Jishe_8/20X40/ar.npy')
    result_df = pd.read_csv('./Doyle/data1.csv')
    tag = np.array(result_df['Output'].to_list())

    # 设置Seaborn风格
    sns.set(style="whitegrid")

    # # 绘制年龄分布直方图
    # plt.figure(figsize=(8, 6))
    # sns.histplot(tag, bins=30, kde=True, color='skyblue')
    # plt.title("DDG Distribution")
    # plt.xlabel("DDG")
    # plt.ylabel("Frequency")
    # plt.show()

    epochs = 200
    batch_size = 128
    root_path = f'/media/data1/Models_ly/3DChemical/doyle-1/generate-100-embbeding/Catalyst_all_predict'
    train_dataset = VAEData(add_spms, lig_spms, base_spms, ar_spms, tag)
    train_predict(epochs=epochs, batch_size=batch_size, lr=1e-3, train_data=train_dataset, root_path=root_path, device=device)