from oo1_cvae_on_doyle_all_final import CVAE_Pre
import os
import random
import torch
import torchvision
import utils
import pandas as pd
import numpy as np
import seaborn as sns
from CGAN import Generator, Discriminator
import torch.nn.functional as F

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

def generate_from_cvae_pre(root_path: str, save_path: str, sample_num: int, latent_dim: int, lig, base, ar, label, device):
    # Load vae model
    cvae = CVAE_Pre(feature_size=800, class_size=100, latent_size=100).to(device)
    # cvae = CVAE(800, 50, 50).to(device)
    utils.load_model(cvae, f"{root_path}/model_weights/cvae/cvae_weights.pth")

    lig_features = np.repeat(lig[np.newaxis, :, :], sample_num, axis=0)
    base_features = np.repeat(base[np.newaxis, :, :], sample_num, axis=0)
    ar_features = np.repeat(ar[np.newaxis, :, :], sample_num, axis=0)

    # sample from the latent space and concat label that you want to generate
    z = torch.randn(sample_num, latent_dim).to(device)
    labels = torch.full(size=(sample_num,), fill_value=label, dtype=torch.int64)
    y = labels.long().to(device)
    lig_features = torch.tensor(lig_features).float().to(device)
    base_features = torch.tensor(base_features).float().to(device)
    ar_features = torch.tensor(ar_features).float().to(device)
    lig_features = lig_features.reshape(lig_features.shape[0], -1)
    base_features = base_features.reshape(base_features.shape[0], -1)
    ar_features = ar_features.reshape(ar_features.shape[0], -1)
    recon = cvae.decode(z, lig_features, base_features, ar_features, y)
    logits = cvae.predict(recon, lig_features, base_features, ar_features)
    recon = recon.detach().cpu().numpy()

    # # 将图像扩展为 4D 张量 (N, C, H, W)，这里假设 C = 1（单通道图像）
    # large_imgs = recon.reshape(recon.shape[0], 1, 20, 40)
    # # 通过双线性插值将图像扩展到 (N, 1, 40, 80)
    # # large_imgs = F.interpolate(large_imgs, size=(40, 80), mode='bilinear', align_corners=False)
    # # 如果想使用最近邻插值
    # large_imgs = F.interpolate(large_imgs, size=(40, 80), mode='nearest')
    # # 如果需要，还可以将结果 reshape 回 (N, 40, 80)
    # large_imgs = large_imgs.squeeze(1)  # (N, 40, 80)
    # large_imgs = large_imgs.detach().cpu().numpy()

    utils.to_heatmap(recon[:9], save_path)

    # utils.to_heatmap(large_imgs[:9], save_path, True)
    print("save:", save_path, "\n")
    # return large_imgs, logits
    return recon, logits

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
    lig_spms = np.load('./Doyle/Jishe_81/20X40/lig.npy')
    add_spms = np.load('./Doyle/Jishe_81/20X40/add.npy')
    base_spms = np.load('./Doyle/Jishe_81/20X40/base.npy')
    ar_spms = np.load('./Doyle/Jishe_81/20X40/ar.npy')

    # lig_spms = np.load('./Doyle/Dis_81/20X40/lig.npy')
    # add_spms = np.load('./Doyle/Dis_81/20X40/add.npy')
    # base_spms = np.load('./Doyle/Dis_81/20X40/base.npy')
    # ar_spms = np.load('./Doyle/Dis_81/20X40/ar.npy')

    # lig_spms = np.load('./npy/ligand20.npy')
    # add_spms = np.load('./npy/additive20.npy')
    # base_spms = np.load('./npy/base20.npy')
    # ar_spms = np.load('./npy/aryl20.npy')

    lig_spms = (lig_spms - lig_spms.min()) / (lig_spms.max() - lig_spms.min())
    add_spms = (add_spms - add_spms.min()) / (add_spms.max() - add_spms.min())
    base_spms = (base_spms - base_spms.min()) / (base_spms.max() - base_spms.min())
    ar_spms = (ar_spms - ar_spms.min()) / (ar_spms.max() - ar_spms.min())


    result_df = pd.read_csv('./Doyle/data1.csv')
    tag = np.array(result_df['Output'].to_list())

    num = lig_spms.shape[0]

    # 设置Seaborn风格
    sns.set(style="whitegrid")

    root_path = f'/media/data1/Models_ly/3DChemical/Doyle-11/generate-100-embbeding/Catalyst_all_predict'

    # root_path = f'/media/data1/Models_ly/3DChemical/Doyle-11/generate-100-distance/Catalyst_all_predict'
    # root_path = f'/media/data1/Models_ly/3DChemical/doyle-1/generate-100-spms/Catalyst_all_predict'
    # compile_index = 1

    all_add_list = []
    all_lig_list = []
    all_base_list = []
    all_ar_list = []
    all_label_list = []

    for compile_index in range(num):
        print(f'++++++++++++++++++++++++++++++++compile_index_{compile_index + 1}++++++++++++++++++++++++++++++++')
        lig_feature = lig_spms[compile_index]
        base_feature = base_spms[compile_index]
        ar_feature = ar_spms[compile_index]

        sample_size = 10

        lig_features_1 = np.repeat(lig_feature[np.newaxis, :, :], sample_size, axis=0)
        base_features_1 = np.repeat(base_feature[np.newaxis, :, :], sample_size, axis=0)
        ar_features_1 = np.repeat(ar_feature[np.newaxis, :, :], sample_size, axis=0)

        all_lig_list.append(lig_features_1)
        all_base_list.append(base_features_1)
        all_ar_list.append(ar_features_1)


        # sample_labels = np.random.uniform(0, 100, sample_size)

        sample_labels = np.ones([1, ]) * tag[compile_index]

        # 定义类别的边界
        edges = generate_edges(100, 0, 100)
        sample_labels_1 = np.digitize(sample_labels, edges)

        utils.make_dir(f"{root_path}/generate_results-3/{compile_index}")

        recon_list = []
        label_list = []
        for i, label in enumerate(sample_labels_1):
            save_path = f"{root_path}/generate_results-3/{compile_index}/{sample_labels[i]}.png"
            recon, logits = generate_from_cvae_pre(root_path=root_path, save_path=save_path, sample_num=sample_size, latent_dim=100,
                                             lig=lig_feature, base=base_feature, ar=ar_feature, label=label, device=device)

            labels = np.ones([sample_size,])*sample_labels[i]

            # print(labels, logits.detach().cpu().numpy())
            # recon_list.append(recon[np.newaxis, :, :])
            recon_list.append(recon)
            label_list.append(labels)
            # label_list.append(logits.detach().cpu().numpy())

        recon_list = np.concatenate(recon_list, axis=0)
        label_list = np.concatenate(label_list, axis=0)

        all_add_list.append(recon_list)
        all_label_list.append(label_list)

    all_add_list = np.concatenate(all_add_list, axis=0)
    all_lig_list = np.concatenate(all_lig_list, axis=0)
    all_base_list = np.concatenate(all_base_list, axis=0)
    all_ar_list = np.concatenate(all_ar_list, axis=0)
    all_label_list = np.concatenate(all_label_list, axis=0)

    # np.save(f"{root_path}/generate_results-3/add.npy", all_add_list.reshape(all_add_list.shape[0], 20, 40))
    # np.save(f"{root_path}/generate_results-3/lig.npy", all_lig_list)
    # np.save(f"{root_path}/generate_results-3/base.npy", all_base_list)
    # np.save(f"{root_path}/generate_results-3/ar.npy", all_ar_list)
    # np.save(f"{root_path}/generate_results-3/label.npy", all_label_list)

    """


    def rmse(imageA, imageB):
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        err = np.sqrt(err)
        return err

    def psnr(imageA, imageB):
        mse_val = rmse(imageA, imageB)
        if mse_val == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse_val))


    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0, '#144C87'), (0.5, 'white'), (1, 'red')])
    def to_heatmap(true, generate, ddg, rmse, psnr, path):
        plt.clf()

        # imgs = imgs.reshape(imgs.shape[0], 20, 40)
        # imgs = imgs.reshape(imgs.shape[0], 40, 80)

        # 创建一个2x5的子图布局
        fig, axes = plt.subplots(1, 2, figsize=(12, 8))

        axes.flat[0].imshow(true, cmap=custom_cmap)
        axes.flat[0].set_title(f"True Image")
        axes.flat[0].axis('off')  # 关闭坐标轴

        axes.flat[1].imshow(generate, cmap=custom_cmap)
        axes.flat[1].set_title(f"Generate Image")
        axes.flat[1].axis('off')  # 关闭坐标轴

        # 调整子图间的间距
        # plt.tight_layout()

        # 添加MSE值
        fig.text(0.5, 0.3, f"DDG: {ddg:.4f}, RMSE: {rmse:.4f}, PSNR: {psnr}", ha='center', fontsize=12)

        # 调整子图间的间距
        plt.subplots_adjust(bottom=0.15)

        # 可选：保存合并的热力图
        plt.savefig(path, dpi=600, bbox_inches='tight')

        # plt.show()

    for imine_smile in imine_smiles_set:
        for thiol_smile in thiol_smiles_set:
            print(f'++++++++++++++++++++++++++++++++compile_index_{compile_index}++++++++++++++++++++++++++++++++')

            # 条件筛选
            filtered_rows = result_df[(result_df['Imine'] == imine_smile) & (result_df['Thiol'] == thiol_smile)]
            cat_index = filtered_rows.index.tolist()
            filtered_cat = filtered_rows['Catalyst'].tolist()
            filtered_ddg = filtered_rows['Output'].tolist()

            imine_index = imine_smiles.index(imine_smile)
            thiol_index = thiol_smiles.index(thiol_smile)
            imine_feature = imine_spms[imine_index]
            thiol_feature = thiol_spms[thiol_index]

            imine_features = np.repeat(imine_feature[np.newaxis, :, :], 43, axis=0)
            thiol_features = np.repeat(thiol_feature[np.newaxis, :, :], 43, axis=0)

            all_imine_list.append(imine_features)
            all_thiol_list.append(thiol_features)

            # 真实cat特征
            true_cat_features = cat_spms[cat_index]

            # # 均匀采样小数数组
            # sample_size = 50
            # sample_labels = np.random.uniform(0, 3, sample_size)
            # 使用原始标签分布
            sample_labels = filtered_ddg

            # 定义类别的边界
            edges = generate_edges(50, -1, 3)
            sample_labels = np.digitize(sample_labels, edges)

            utils.make_dir(f"{root_path}/generate_results/{compile_index}")
            # utils.make_dir(f"{root_path}/generate_results/{imine_smile}/{thiol_smile}")
            with open(f"{root_path}/generate_results/{compile_index}/compile.txt", "w") as f:
                f.write(f'Imine: {imine_smile}, Thiol: {thiol_smile}')

            recon_list = []
            label_list = []
            for i, label in enumerate(sample_labels):
                label_1 = filtered_ddg[i]
                save_path = f"{root_path}/generate_results/{compile_index}/{label_1}.png"
                # recon = generate_from_cvae(root_path=root_path, save_path=save_path, sample_num=1, latent_dim=10,
                #                            imine=imine_feature, thiol=thiol_feature, label=label, device=device)
                # recon = generate_from_cvae(root_path=root_path, save_path=save_path, sample_num=1, latent_dim=50, imine=imine_feature, thiol=thiol_feature, label=label, device=device)
                recon = generate_from_cvae_smile(root_path=root_path, save_path=save_path, sample_num=1, latent_dim=100,
                                           imine=imine_smile, thiol=thiol_smile, label=label, device=device)
                # recon = generate_from_gan(root_path=root_path, save_path=save_path, sample_num=1, latent_dim=100,
                #                                  imine=imine_smile, thiol=thiol_smile, label=label, device=device)
                labels = np.ones([1, ]) * label_1
                recon_list.append(recon[np.newaxis, :, :])
                label_list.append(labels)
                generate_psnr = psnr(true_cat_features[i], recon.reshape(-1).reshape(20, 40))
                generate_rmse = rmse(true_cat_features[i], recon.reshape(-1).reshape(20, 40))
                print(f'Imine: {imine_smile}, Thiol: {thiol_smile}, DDG: {label_1}, generate_psnr: {generate_psnr}, generate_rmse: {generate_rmse}')
                # to_heatmap(true_cat_features[i], recon.reshape(-1).reshape(20, 40), label_1, generate_rmse, generate_psnr, save_path)

            recon_list = np.concatenate(recon_list, axis=0)
            label_list = np.concatenate(label_list, axis=0)

            all_cat_list.append(recon_list)
            all_label_list.append(label_list)

            compile_index += 1

    all_cat_list = np.concatenate(all_cat_list, axis=0)
    all_imine_list = np.concatenate(all_imine_list, axis=0)
    all_thiol_list = np.concatenate(all_thiol_list, axis=0)
    all_label_list = np.concatenate(all_label_list, axis=0)

    # np.save(f"{root_path}/generate_results/cat2.npy", all_cat_list)
    np.save(f"{root_path}/generate_results/cat2.npy", all_cat_list.reshape(all_cat_list.shape[0], 20, 40))
    np.save(f"{root_path}/generate_results/imine2.npy", all_imine_list)
    np.save(f"{root_path}/generate_results/thiol2.npy", all_thiol_list)
    np.save(f"{root_path}/generate_results/label2.npy", all_label_list)
    """