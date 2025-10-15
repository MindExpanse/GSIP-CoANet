
import datetime  # 导入处理日期时间的模块
import os.path as osp
import cv2
import numpy as np
import pandas  # 导入用于数据分析的模块
import xlwt  # 导入用于操作Excel文件的模块
import matplotlib.pyplot as plt  # 导入用于绘图的模块
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split  # 导入用于数据集划分的函数
from torchvision import transforms
import matplotlib.cm as cm
from utils2 import set_logging_config  # 导入自定义的日志设置函数
from dataloader import SPMSData  # 导入自定义的数据加载类
import torch  # 导入PyTorch深度学习框架
import os  # 导入操作系统相关的模块
import random  # 导入随机数生成模块
import logging  # 导入日志记录模块
import argparse  # 导入解析命令行参数的模块
import importlib  # 导入动态加载模块的模块
from trainer import Trainer  # 导入自定义的训练器类
from metric import *  # 导入自定义的评估指标函数
from resnet import resnet18, resnet34, CNN2Large  # 导入自定义的深度学习模型
from vit import ViT  # 导入自定义的Vision Transformer模型
from Danet import DAnet, DAnet2  # 导入自定义的模型
from torch.utils.data import WeightedRandomSampler, DataLoader  # 导入PyTorch中用于数据加载的类和函数
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score  # 导入评估指标函数
from sklearn.model_selection import KFold  # 导入用于K折交叉验证的类
from cam_utils import GradCAM, show_cam_on_image, center_crop_img
import pandas as pd
from torchvision import models, transforms
import torch.nn.functional as F
import torch
import random
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 可视化对二维散点集合进行网格划分的矩阵也就是极射赤平投影描述符
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

import seaborn as sns

cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0, 'blue'), (1, 'red')])
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0, '#144C87'), (0.5, 'white'), (1, 'red')])

from grad_cam import (
    BackPropagation,
    Deconvnet,
    # GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images


def get_classtable():
    classes = []
    with open("samples/synset_words.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes


def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def save_gradient(filename, gradient):

    gradient -= gradient.min()
    gradient /= gradient.max()

    plt.clf()
    sns.heatmap(gradient[..., 0], cbar=True, cmap=custom_cmap)
    # Set a higher resolution (e.g., 300 dots per inch)
    plt.savefig(filename, dpi=900, bbox_inches='tight')
    plt.show()

    # gradient *= 255.0
    # cv2.imwrite(filename, np.uint8(gradient[..., 0]))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):

    # gcam = gcam.cpu().numpy()
    # heatmap = cv2.applyColorMap(np.uint8(255 * gcam), cv2.COLORMAP_JET)
    # use_rgb = False
    # if use_rgb:
    #     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # heatmap = np.float32(heatmap) / 255
    #
    # if np.max(raw_image) > 1:
    #     raise Exception(
    #         "The input image should np.float32 in the range [0, 1]")
    #
    # cam = heatmap + raw_image
    # cam = cam / np.max(cam)
    #
    # plt.imshow(cam)
    # plt.show()
    #
    # cv2.imwrite(filename, cam)


    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3]
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        # gcam = (cmap.astype(float) + raw_image.astype(float)) / 2
        gcam = cmap.astype(float)

    plt.clf()
    sns.heatmap(gcam[..., 0], cbar=True, cmap=custom_cmap)
    # Set a higher resolution (e.g., 300 dots per inch)
    plt.savefig(filename, dpi=900, bbox_inches='tight')
    plt.show()

    # plt.imshow(gcam)
    # plt.show()
    # cv2.imwrite(filename, np.uint8(gcam[..., 0]))


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    # maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (40, 20), interpolation=cv2.INTER_NEAREST)
    plt.clf()
    plt.imshow(maps)
    plt.show()
    # cv2.imwrite(filename, maps)

    # plt.clf()
    # sns.heatmap(maps[..., 0], cbar=True, cmap=custom_cmap)
    # # Set a higher resolution (e.g., 300 dots per inch)
    # plt.savefig(filename, dpi=900, bbox_inches='tight')
    # plt.show()


# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Make input tensor require gradient
    X.requires_grad_()

    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    model.zero_grad()
    loss_fun = torch.nn.MSELoss(reduction='mean')
    y_pred, _ = model(X)
    loss = loss_fun(y_pred, y)
    loss.backward()
    saliency2 = X.grad.abs()[:, 1, :]
    saliency,_ = X.grad.abs().max(axis=1)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    # return saliency
    return saliency2

def show_saliency_maps(X, y, model):
    # Convert X and y from numpy arrays to Torch Tensors
    # X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
    X_tensor = X
    y_tensor = torch.FloatTensor(y)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.detach().numpy()
    X = X.detach().numpy()
    N = X.shape[0]
    for i in range(N):
        plt.subplot(2, N, i + 1)
        # plt.imshow(X[i].transpose(1, 2, 0))
        plt.imshow(X[i][0,:].squeeze(), cmap=plt.cm.hot)
        plt.axis('off')
        plt.title(y[i])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()

def main(args_opt):
    # 设置随机种子，保证结果可重复
    np.random.seed(args_opt.seed)
    torch.manual_seed(args_opt.seed)
    torch.cuda.manual_seed_all(args_opt.seed)
    random.seed(args_opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 根据参数设置实验名称
    args_opt.exp_name = 'EP-{}_BS-{}_LR-{}_SPMS'.format(args_opt.epoch, args_opt.batch_size, args_opt.learning_rate)

    # 设置模型权重保存的路径
    # args_opt.checkpoint_dir = os.path.join(args_opt.path_root, args_opt.exp_name, 'fold' + str(1), 'checkpoints/')
    args_opt.checkpoint_dir = os.path.join(args_opt.path_root, args_opt.exp_name, 'CNN',
                                           'fold_' + str(0) + '_run_' + str(1), 'checkpoints/')

    result_test = []  # 存储测试结果
    args_opt.save_test_result = os.path.join(args_opt.path_root, args_opt.exp_name, 'test_result-1')  # 设置保存测试结果的路径
    workbook = xlwt.Workbook(encoding='utf-8')  # 创建一个Excel文件

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # 加载生成的数据
    generate_path = '/media/data1/Models_ly/3DChemical/Doyle-11/generate-100-embbeding/Catalyst_all_predict/generate_results-3'
    generate_lig_spms = np.load(f'{generate_path}/lig.npy')
    generate_add_spms = np.load(f'{generate_path}/add.npy')
    generate_base_spms = np.load(f'{generate_path}/base.npy')
    generate_ar_spms = np.load(f'{generate_path}/ar.npy')
    generate_tag = np.load(f'{generate_path}/label.npy').reshape(-1)

    # 数据标准化
    generate_lig_spms_std = (generate_lig_spms - generate_lig_spms.min()) / (
            generate_lig_spms.max() - generate_lig_spms.min())
    generate_add_spms_std = (generate_add_spms - generate_add_spms.min()) / (
            generate_add_spms.max() - generate_add_spms.min())
    generate_base_spms_std = (generate_base_spms - generate_base_spms.min()) / (
            generate_base_spms.max() - generate_base_spms.min())
    generate_ar_spms_std = (generate_ar_spms - generate_ar_spms.min()) / (
            generate_ar_spms.max() - generate_ar_spms.min())
    # generate_react_spms_std = np.concatenate([generate_lig_spms_std.reshape(3955*3, 40, 80, 1),
    #                                           generate_add_spms_std.reshape(3955*3, 40, 80, 1),
    #                                           generate_base_spms_std.reshape(3955*3, 40, 80, 1),
    #                                           generate_ar_spms_std.reshape(3955*3, 40, 80, 1)],
    #                                          axis=3)
    generate_react_spms_std = np.concatenate([generate_lig_spms_std.reshape(3955, 20, 40, 1),
                                              generate_add_spms_std.reshape(3955, 20, 40, 1),
                                              generate_base_spms_std.reshape(3955, 20, 40, 1),
                                              generate_ar_spms_std.reshape(3955, 20, 40, 1)],
                                             axis=3)

    generate_tag_scale = generate_tag.max() - generate_tag.min()  # 计算标签的尺度
    generate_tag_min = generate_tag.min()  # 记录标签的最小值
    generate_tag_std = (generate_tag - generate_tag_min) / generate_tag_scale  # 标签标准化

    # 加载数据
    total_lig_spms = np.load('./Doyle/Jishe_81/20X40/lig.npy')
    total_add_spms = np.load('./Doyle/Jishe_81/20X40/add.npy')
    total_base_spms = np.load('./Doyle/Jishe_81/20X40/base.npy')
    total_ar_spms = np.load('./Doyle/Jishe_81/20X40/ar.npy')
    result_df = pd.read_csv('./Doyle/data1.csv')
    tag = np.array(result_df['Output'].to_list())

    print(total_lig_spms.shape, total_add_spms.shape, total_base_spms.shape, total_ar_spms.shape)

    # 数据标准化
    lig_spms_std = (total_lig_spms - total_lig_spms.min()) / (total_lig_spms.max() - total_lig_spms.min())
    add_spms_std = (total_add_spms - total_add_spms.min()) / (total_add_spms.max() - total_add_spms.min())
    base_spms_std = (total_base_spms - total_base_spms.min()) / (total_base_spms.max() - total_base_spms.min())
    ar_spms_std = (total_ar_spms - total_ar_spms.min()) / (total_ar_spms.max() - total_ar_spms.min())
    react_spms_std = np.concatenate([lig_spms_std.reshape(3955, 20, 40, 1), add_spms_std.reshape(3955, 20, 40, 1),
                                     base_spms_std.reshape(3955, 20, 40, 1), ar_spms_std.reshape(3955, 20, 40, 1),
                                     ], axis=3)
    # react_spms_std = np.concatenate([lig_spms_std.reshape(3955, 40, 80, 1), add_spms_std.reshape(3955, 40, 80, 1),
    #                                  base_spms_std.reshape(3955, 40, 80, 1), ar_spms_std.reshape(3955, 40, 80, 1),
    #                                  ], axis=3)
    tag_scale = tag.max() - tag.min()
    tag_min = tag.min()
    tag_std = (tag - tag_min) / tag_scale

    device = get_device(True)

    # 将图像展平成一维数组
    X_flat = react_spms_std.reshape(react_spms_std.shape[0], -1)  # 将图像展平成 (n_samples, 50*100*3)
    generate_X_flat = generate_react_spms_std.reshape(generate_react_spms_std.shape[0],
                                                      -1)  # 将图像展平成 (n_samples, 50*100*3)

    # 随机划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_flat, tag_std, test_size=0.3, random_state=42)

    X_train = np.concatenate([X_train, generate_X_flat], axis=0)
    y_train = np.concatenate([y_train, generate_tag_std], axis=0)

    """# 使用随机森林回归模型，也可以选择其他回归模型
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    predict = rf.predict(X_test)
    predict_ddG = predict * tag_scale + tag_min
    predict_ddG = predict_ddG.reshape(-1, )
    truth_ddG = y_test * tag_scale + tag_min
    truth_ddG = truth_ddG.reshape(-1, )

    r2 = r2_score(predict_ddG, truth_ddG)
    mae = mean_absolute_error(predict_ddG, truth_ddG)
    # 计算MSE
    mse = mean_squared_error(predict_ddG, truth_ddG)
    # 计算RMSE
    rmse = np.sqrt(mse)

    # r2_list.append(r2)
    # mae_list.append(mae)
    # rmse_list.append(rmse)
    print('DDG R2: %f, MAE: %f, RMSE: %f' % (r2, mae, rmse))

    feature_importances = rf.feature_importances_

    # 假设importances_cat是你的二维NumPy矩阵
    min_value = feature_importances.min()
    max_value = feature_importances.max()

    # 将数据缩放到[0, 1]范围内
    feature_importances = (feature_importances - min_value) / (max_value - min_value)
    _, height, width, channel = react_spms_std.shape
    importances = feature_importances.reshape(height, width, channel)"""

    output_dir = os.path.join(args_opt.save_test_result, 'results2D')
    os.makedirs(output_dir, exist_ok=True)

    # np.save(osp.join(output_dir, 'rf_importance.npy'), importances)
    importances = np.load(osp.join(output_dir, 'rf_importance.npy'))

    def make_model():
        model = CNN2Large()  # 创建自定义的卷积神经网络模型
        assert os.path.exists(os.path.join(args_opt.checkpoint_dir, 'checkpoint.pth.tar')), '指定模型文件未找到，请检查'
        best_checkpoint = torch.load(os.path.join(args_opt.checkpoint_dir, 'model_best.pth.tar'),
                                     map_location=torch.device('cpu'))
        try:
            # 加载模型的权重
            model.load_state_dict(best_checkpoint['enc_module_state_dict'])
        except:
            new_enc_module_state_dict = {f'module.{k}': v for k, v in
                                         best_checkpoint['enc_module_state_dict'].items()}
            model.load_state_dict(new_enc_module_state_dict)
        model.to(device)
        model.eval()

        return model


    target_layer = 'conv2'
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # patche_sizes = [10, 15, 25, 35, 45, 90]
    patche_sizes = [5]
    stride = 1
    n_batches = 128

    topk = 1

    result_list = []
    result_list_1 = []
    for index, row in result_df.iterrows():
        lig = row['Ligand']
        add = row['Additive']
        base = row['Base']
        ar = row['Aryl halide']
        print(f"+++++++++++++++Index: {index}+++++++++++++++")
        print(f"Ligand: {lig}, Additive: {add}, Base: {base}, Aryl halide: {ar}")
        cat_index = [index]
        img = react_spms_std[cat_index]
        img = img.squeeze(0)
        label = np.array([tag[cat_index]])
        label_std = np.array([tag_std[cat_index]])
        img_tensor = data_transform(img)
        input_tensor = torch.unsqueeze(img_tensor, dim=0).float()
        input_tensor = input_tensor.to(device)

        torch.set_grad_enabled(True)

        model = make_model()
        target_layers = [model.conv2]  # 选择最后一个卷积层

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        target_category = 0  # 根据需要设定类别
        regions = cam(input_tensor=input_tensor, target_category=target_category)
        # grayscale_cam = regions[0, :]
        # visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        # plt.imshow(visualization[:, :, :])
        # plt.show()
        regions = torch.tensor(regions, device=device).unsqueeze(1)

        model = make_model()

        # =========================================================================
        print("Vanilla Backpropagation:")
        bp = BackPropagation(model=model)
        probs, ids = bp.forward(input_tensor)  # sorted

        bp.backward(ids=ids[:, [0]])
        gradients_1 = bp.generate()

        result_map = gradients_1.cpu().numpy().transpose(0, 2, 3, 1)
        result_map = np.abs(result_map)
        result_list_1.append(result_map[0])
        result_map = (result_map - result_map.min()) / (result_map.max() - result_map.min())
        atten = 0.5 * result_map[0] + 0.5 * importances
        result_list.append(atten)

        # Save results as image files
        print("\t#{}: {} ({:.5f}, {:.5f})".format(0, label[0, 0], label_std[0, 0], probs[0, 0]))

        save_gradient(
            filename=osp.join(
                output_dir,
                "{}-vanilla-{}.png".format(index, label[0, 0]),
            ),
            gradient=atten,
        )

        # Remove all the hook function in the "model"
        bp.remove_hook()

    result_list_1 = np.array(result_list_1)
    result_list = np.array(result_list)
    np.save(osp.join(output_dir, 'vanilla_all.npy'), result_list_1)
    np.save(osp.join(output_dir, 'ensemble_all.npy'), result_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or eval')  # 模式：训练或评估
    parser.add_argument('--resume', type=bool, default=False)  # 是否恢复训练
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--gpu_ids', type=list, default=[0], help='number of gpu')  # 使用的GPU编号
    parser.add_argument('--cross_num',type=int, default=10)  # 交叉验证的折数
    parser.add_argument('--k_fold', type=bool, default=False)  # 是否使用K折交叉验证
    parser.add_argument('--plot_flag', type=bool, default=True)  # 是否绘制图表
    parser.add_argument('--learning_rate', type=int, default=1e-3)  # 学习率
    parser.add_argument('--weight_decay', type=int, default=1e-5)  # 权重衰减
    parser.add_argument('--batch_size', type=int, default=128)  # 批量大小
    parser.add_argument('--epoch', type=int, default=200)  # 迭代轮数
    parser.add_argument('--stop_num', type=int, default=200)  # 停止训练的步数
    parser.add_argument('--num_workers', type=int, default=0)  # 数据加载器的工作进程数
    parser.add_argument('--seed', type=int, default=222, help='random seed')  # 随机种子
    # parser.add_argument('--path_root', type=str,
    #                     default='/media/data1/Models_ly/3DChemical/denmark-1/spms-cnn-cat2-channel-20X40-generate-100-embbeding-retrainAllCPA-train-gen-smile/',
    #                     help='path that checkpoint and logs will be saved and loaded. '
    #                          'It is assumed that the checkpoint file is placed under the directory ./checkpoints')  # 模型和日志保存路径
    parser.add_argument('--path_root', type=str,
                        # default='/media/data1/Models_ly/3DChemical/denmark-2/spms-cnn-cat2-channel-20X40/',
                        # default='/media/data1/Models_ly/3DChemical/denmark-2/spms-cnn-cat2-channel-20X40-generate-100-embbeding-retrainAllCPA-train-gen-predict-all-1/',
                        # default='/media/data1/Models_ly/3DChemical/doyle-11/spms-cnn-cat2-channel-20X40-generate-100-embbeding-retrainAllCPA-train-gen-predict-all-5Large/',
                        default='/media/data1/Models_ly/3DChemical/Doyle-11/ensemble-spms-cnn-cat2-channel-20X40-all-1/',
                        help='path that checkpoint and logs will be saved and loaded. '
                             'It is assumed that the checkpoint file is placed under the directory ./checkpoints')  # 模型和日志保存路径
    parser.add_argument('--display_step', type=int, default=1, help='display training information in how many step')  # 显示训练信息的步数间隔
    parser.add_argument('--log_step', type=int, default=1, help='log information in how many steps')  # 记录日志信息的步数间隔
    parser.add_argument('--interval', type=int, default=1, help='log information in how many vals')  # 记录日志信息的值间隔

    args_opt = parser.parse_args()

    # 训练
    main(args_opt)
    #
    # args_opt.mode = 'eval'  # 设置模式为评估
    # # 调用主函数
    # main(args_opt)
