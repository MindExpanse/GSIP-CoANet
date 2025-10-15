from cvae_on_denmark_all import CVAE, CVAE_smile, CVAE_Pre, rdkit_fp_calc
import os
import random
import torch
import torchvision
import utils
import pandas as pd
import numpy as np
import seaborn as sns
from CGAN import Generator, Discriminator

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

def generate_from_gan(root_path: str, save_path: str, sample_num: int, latent_dim: int, imine, thiol, label, device):
    # Load cgan model
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    generator.eval()
    discriminator.eval()
    utils.load_model(generator, f"{root_path}/weights/generator_last.pkl")
    utils.load_model(discriminator, f"{root_path}/weights/discriminator_last.pkl")

    z = torch.tensor(np.random.normal(0, 1, (sample_num, latent_dim))).float().to(device)
    labels = torch.full(size=(sample_num,), fill_value=label, dtype=torch.int64)
    y = labels.long().to(device)
    imine_features = np.array([imine])
    thiol_features = np.array([thiol])
    imine_features = np.repeat(imine_features, sample_num, axis=0)
    thiol_features = np.repeat(thiol_features, sample_num, axis=0)
    # RDKit Fingerprint: 11 types
    # fp_type = ['Avalon', 'AtomPaires', 'TopologicalTorsions', 'MACCSKeys', 'RDKit', 'RDKitLinear', 'LayeredFingerprint', 'Morgan', 'FeaturedMorgan', "Estate", "EstateIndices"]:
    imine_features = rdkit_fp_calc(smiles=imine_features, is_sdf=False, fp_type='RDKit', fp_length=32).float().to(device)
    thiol_features = rdkit_fp_calc(smiles=thiol_features, is_sdf=False, fp_type='RDKit', fp_length=32).float().to(device)
    gen_imgs = generator(z, y, imine_features, thiol_features)

    # img_array = gen_imgs.data.reshape((gen_imgs.shape[2], -1))
    img_array = gen_imgs.data.reshape((gen_imgs.shape[0], 20, 40))

    recon = img_array.detach().cpu().numpy()
    # torchvision.utils.save_image(imgs, path, nrow=10)
    # utils.to_heatmap(recon[:9], save_path)
    print("save:", save_path, "\n")
    return recon

def generate_from_cvae(root_path: str, save_path: str, sample_num: int, latent_dim: int, imine, thiol, label, device):
    # Load vae model
    cvae = CVAE(800, 10, 10).to(device)
    # cvae = CVAE(800, 50, 50).to(device)
    utils.load_model(cvae, f"{root_path}/model_weights/cvae/cvae_weights.pth")

    imine_features = np.repeat(imine[np.newaxis, :, :], sample_num, axis=0)
    thiol_features = np.repeat(thiol[np.newaxis, :, :], sample_num, axis=0)

    # sample from the latent space and concat label that you want to generate
    z = torch.randn(sample_num, latent_dim).to(device)
    labels = torch.full(size=(sample_num,), fill_value=label, dtype=torch.int64)
    y = labels.float().to(device)
    imine_features = torch.tensor(imine_features).float().to(device)
    thiol_features = torch.tensor(thiol_features).float().to(device)
    imine_features = imine_features.reshape(imine_features.shape[0], -1)
    thiol_features = thiol_features.reshape(thiol_features.shape[0], -1)
    recon = cvae.decode(z, imine_features, thiol_features, y).detach().cpu().numpy()
    # torchvision.utils.save_image(imgs, path, nrow=10)
    # utils.to_heatmap(recon[:9], save_path)
    # print("save:", save_path, "\n")
    return recon


def generate_from_cvae_pre(root_path: str, save_path: str, sample_num: int, latent_dim: int, imine, thiol, label, device):
    # Load vae model
    cvae = CVAE_Pre(800, 50, 100).to(device)
    # cvae = CVAE(800, 50, 50).to(device)
    utils.load_model(cvae, f"{root_path}/model_weights/cvae/cvae_weights.pth")

    imine_features = np.repeat(imine[np.newaxis, :, :], sample_num, axis=0)
    thiol_features = np.repeat(thiol[np.newaxis, :, :], sample_num, axis=0)

    # sample from the latent space and concat label that you want to generate
    z = torch.randn(sample_num, latent_dim).to(device)
    labels = torch.full(size=(sample_num,), fill_value=label, dtype=torch.int64)
    y = labels.long().to(device)
    imine_features = torch.tensor(imine_features).float().to(device)
    thiol_features = torch.tensor(thiol_features).float().to(device)
    imine_features = imine_features.reshape(imine_features.shape[0], -1)
    thiol_features = thiol_features.reshape(thiol_features.shape[0], -1)
    recon = cvae.decode(z, imine_features, thiol_features, y)
    logits = cvae.predict(recon, imine_features, thiol_features)
    recon = recon.detach().cpu().numpy()
    # utils.to_heatmap(recon[:9], save_path)
    # print("save:", save_path, "\n")
    return recon, logits


def generate_from_cvae_smile(root_path: str, save_path: str, sample_num: int, latent_dim: int, imine, thiol, label, device):
    # Load vae model
    cvae = CVAE_smile(800, 32, 100).to(device)
    # cvae = CVAE(800, 10, 10).to(device)
    # cvae = CVAE(800, 50, 50).to(device)
    utils.load_model(cvae, f"{root_path}/model_weights/cvae/cvae_weights.pth")

    imine_features = np.array([imine])
    thiol_features = np.array([thiol])
    imine_features = np.repeat(imine_features, sample_num, axis=0)
    thiol_features = np.repeat(thiol_features, sample_num, axis=0)

    # imine_features = np.repeat(imine, sample_num, axis=0)
    # thiol_features = np.repeat(thiol, sample_num, axis=0)

    # sample from the latent space and concat label that you want to generate
    z = torch.randn(sample_num, latent_dim).to(device)
    labels = torch.full(size=(sample_num,), fill_value=label, dtype=torch.int64)
    y = labels.long().to(device)

    # RDKit Fingerprint: 11 types
    # fp_type = ['Avalon', 'AtomPaires', 'TopologicalTorsions', 'MACCSKeys', 'RDKit', 'RDKitLinear', 'LayeredFingerprint', 'Morgan', 'FeaturedMorgan', "Estate", "EstateIndices"]:
    imine_features = rdkit_fp_calc(smiles=imine_features, is_sdf=False, fp_type='RDKit', fp_length=32).float().to(device)
    thiol_features = rdkit_fp_calc(smiles=thiol_features, is_sdf=False, fp_type='RDKit', fp_length=32).float().to(device)

    recon = cvae.decode(z, imine_features, thiol_features, y).detach().cpu().numpy()
    # torchvision.utils.save_image(imgs, path, nrow=10)
    # utils.to_heatmap(recon[:9], save_path)
    # print("save:", save_path, "\n")
    return recon

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
    # cat_spms = np.load('Jishe_new_81/20X40/cat.npy')
    # imine_spms = np.load('Jishe_new_81/20X40/imine.npy')
    # thiol_spms = np.load('Jishe_new_81/20X40/thiol.npy')

    # cat_spms = np.load('Dis_new_81_1/20X40/cat.npy')
    # imine_spms = np.load('Dis_new_81_1/20X40/imine.npy')
    # thiol_spms = np.load('Dis_new_81_1/20X40/thiol.npy')

    cat_spms = np.load('SPMS_20/cat.npy')
    imine_spms = np.load('SPMS_20/imine.npy')
    thiol_spms = np.load('SPMS_20/thiol.npy')
    cat_spms = (cat_spms - cat_spms.min()) / (cat_spms.max() - cat_spms.min())
    imine_spms = (imine_spms - imine_spms.min()) / (imine_spms.max() - imine_spms.min())
    thiol_spms = (thiol_spms - thiol_spms.min()) / (thiol_spms.max() - thiol_spms.min())

    # cat_spms = np.load('SPMS_20/cat.npy')
    # imine_spms = np.load('SPMS_20/imine.npy')
    # thiol_spms = np.load('SPMS_20/thiol.npy')

    tag = np.load('./Reaction_Result/ddG.npy')

    # 设置Seaborn风格
    sns.set(style="whitegrid")

    # print(cat_ddg)
    # root_path = f'/media/data1/Models_ly/3DChemical/denmark-1/generate-100-embbeding/Catalyst_all_smile_cgan_3'
    # root_path = f'/media/data1/Models_ly/3DChemical/denmark-1/generate-100-embbeding/Catalyst_all_smile_4'
    # root_path = f'/media/data1/Models_ly/3DChemical/denmark-1/generate-100-embbeding/Catalyst_all_predict'

    # root_path = f'/media/data1/Models_ly/3DChemical/denmark-1/generate-100-distance/Catalyst_all_predict'
    root_path = f'/media/data1/Models_ly/3DChemical/denmark-1/generate-100-spms/Catalyst_all_predict'
    os.makedirs(os.path.join(root_path, "generate_results"), exist_ok=True)

    compile_index = 1

    all_cat_list = []
    all_imine_list = []
    all_thiol_list = []
    all_label_list = []


    for imine_smile in imine_smiles_set:
        for thiol_smile in thiol_smiles_set:
            print(f'++++++++++++++++++++++++++++++++compile_index_{compile_index}++++++++++++++++++++++++++++++++')
            print("imine_smile:", imine_smile)
            print("thiol_smile:", thiol_smile)
            compile_index += 1
            imine_index = imine_smiles.index(imine_smile)
            thiol_index = thiol_smiles.index(thiol_smile)
            imine_feature = imine_spms[imine_index]
            thiol_feature = thiol_spms[thiol_index]

            imine_features = np.repeat(imine_feature[np.newaxis, :, :], 50, axis=0)
            thiol_features = np.repeat(thiol_feature[np.newaxis, :, :], 50, axis=0)

            all_imine_list.append(imine_features)
            all_thiol_list.append(thiol_features)

            # 均匀采样小数数组
            sample_size = 50
            sample_labels = np.random.uniform(0, 3, sample_size)

            # 定义类别的边界
            edges = generate_edges(50, -1, 3)
            sample_labels_1 = np.digitize(sample_labels, edges)

            utils.make_dir(f"{root_path}/generate_results_1/{compile_index}")

            recon_list = []
            label_list = []
            for i, label in enumerate(sample_labels_1):
                save_path = f"{root_path}/generate_results_1/{compile_index}/{sample_labels[i]}.png"
                # recon = generate_from_cvae(root_path=root_path, save_path=save_path, sample_num=1, latent_dim=10, imine=imine_feature, thiol=thiol_feature, label=label, device=device)
                # recon = generate_from_cvae(root_path=root_path, save_path=save_path, sample_num=1, latent_dim=50, imine=imine_feature, thiol=thiol_feature, label=label, device=device)
                # recon = generate_from_cvae_smile(root_path=root_path, save_path=save_path, sample_num=1, latent_dim=100,
                #                            imine=imine_smile, thiol=thiol_smile, label=label, device=device)
                recon, logits = generate_from_cvae_pre(root_path=root_path, save_path=save_path, sample_num=1, latent_dim=100,
                                                 imine=imine_feature, thiol=thiol_feature, label=label, device=device)
                # recon = generate_from_gan(root_path=root_path, save_path=save_path, sample_num=10, latent_dim=100,
                #                           imine=imine_smile, thiol=thiol_smile, label=label, device=device)

                labels = np.ones([1,])*sample_labels[i]

                print(labels, logits.detach().cpu().numpy())
                recon_list.append(recon[np.newaxis, :, :])
                # label_list.append(labels)
                label_list.append(logits.detach().cpu().numpy())

            recon_list = np.concatenate(recon_list, axis=0)
            label_list = np.concatenate(label_list, axis=0)

            all_cat_list.append(recon_list)
            all_label_list.append(label_list)

            compile_index += 1

    all_cat_list = np.concatenate(all_cat_list, axis=0)
    all_imine_list = np.concatenate(all_imine_list, axis=0)
    all_thiol_list = np.concatenate(all_thiol_list, axis=0)
    all_label_list = np.concatenate(all_label_list, axis=0)

    np.save(f"{root_path}/generate_results/cat.npy", all_cat_list.reshape(all_cat_list.shape[0], 20, 40))
    # np.save(f"{root_path}/generate_results/cat.npy", all_cat_list)
    np.save(f"{root_path}/generate_results/imine.npy", all_imine_list)
    np.save(f"{root_path}/generate_results/thiol.npy", all_thiol_list)
    np.save(f"{root_path}/generate_results/label.npy", all_label_list)

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