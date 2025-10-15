import datetime
import json
import time

import joblib
import lightgbm
import numpy as np
import pandas
import pandas as pd
import xlwt
from matplotlib import pyplot as plt
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from utils2 import set_logging_config
from dataloader import SPMSData
import torch
import os
import random
import logging
import argparse
import importlib
from trainer2 import Trainer
from metric import *
from resnet import resnet18, resnet34, CNN
from vit import ViT
from Danet import DAnet, DAnet2
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,BaggingRegressor,BaggingRegressor,ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score
import xgboost as xgb  # Import XGBoost
from catboost import CatBoostRegressor  # Import CatBoost

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib


def main(args_opt):

    # set random seed
    np.random.seed(args_opt.seed)
    torch.manual_seed(args_opt.seed)
    torch.cuda.manual_seed_all(args_opt.seed)
    random.seed(args_opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args_opt.exp_name = 'EP-{}_BS-{}_LR-{}_SPMS'.format(args_opt.epoch, args_opt.batch_size, args_opt.learning_rate)

    result_test = []
    args_opt.save_test_result = os.path.join(args_opt.path_root, args_opt.exp_name, 'test_result-cnn-rf-11')
    workbook = xlwt.Workbook(encoding='utf-8')
    workbook2 = xlwt.Workbook(encoding='utf-8')
    for fold in range(args_opt.folds):
        total_predict_ddG = []
        total_test_ddG = []
        total_hist = []
        r2_list = []
        mae_list = []
        rmse_list = []
        for i in range(1, args_opt.cross_num+1):
        # for fold in range(2, 11):
        # for fold in [3]:
            if args_opt.mode == 'train':
                args_opt.log_dir = os.path.join(args_opt.path_root, args_opt.exp_name, 'CNN', 'fold_'+str(fold)+'_run_'+str(i),
                                                'logs/{}/'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
            else:
                args_opt.log_dir = os.path.join(args_opt.save_test_result, 'CNN',
                                                'fold_'+str(fold)+'_run_'+str(i),
                                                'logs/{}/'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))

            args_opt.checkpoint_dir = os.path.join(args_opt.path_root, args_opt.exp_name, 'CNN', 'fold_'+str(fold)+'_run_'+str(i), 'checkpoints/')
            # args_opt.checkpoint_dir = os.path.join(args_opt.path_root, args_opt.exp_name, 'CNN', 'fold'+str(i), 'checkpoints/')

            set_logging_config(args_opt.log_dir)

            logger = logging.getLogger('{}-run{}'.format(args_opt.mode, i))

            # Load the configuration params of the experiment
            # logger.info('Launching experiment from: {}'.format(config_file))
            logger.info('Generated logs will be saved to: {}'.format(args_opt.log_dir))
            logger.info('Generated checkpoints will be saved to: {}'.format(args_opt.checkpoint_dir))
            print()

            logger.info('-------------command line arguments-------------')
            logger.info(args_opt)

            enc_module = CNN()

            # multi-gpu configuration
            # 在获取多GPU信息时，需要先声明
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
            # if len(args_opt.gpu_ids) >= 2:
            #     gpu_str = str(args_opt.gpu_ids[0])
            #     for i in range(1, len(args_opt.gpu_ids)):
            #         gpu_str += ',' + str(args_opt.gpu_ids[i])
            #     os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
            # else:
            #     os.environ['CUDA_VISIBLE_DEVICES'] = str(args_opt.gpu_ids[0])
            [print('GPU: {}  Spec: {}'.format(i, torch.cuda.get_device_name(i))) for i in args_opt.gpu_ids]

            # 加载生成的数据
            generate_path = '/media/data1/Models_ly/3DChemical/denmark-1/generate-100-embbeding/Catalyst_all_predict/generate_results'
            # generate_path = '/media/data1/Models_ly/3DChemical/denmark-1/generate-100-embbeding/Catalyst_all_smile_cgan_3/generate_results'
            generate_cat_spms = np.load(f'{generate_path}/cat.npy')
            generate_imine_spms = np.load(f'{generate_path}/imine.npy')
            generate_thiol_spms = np.load(f'{generate_path}/thiol.npy')
            generate_tag = np.load(f'{generate_path}/label.npy').reshape(-1)

            # 数据标准化
            generate_cat_spms_std = (generate_cat_spms - generate_cat_spms.min()) / (
                    generate_cat_spms.max() - generate_cat_spms.min())
            generate_imine_spms_std = (generate_imine_spms - generate_imine_spms.min()) / (
                    generate_imine_spms.max() - generate_imine_spms.min())
            generate_thiol_spms_std = (generate_thiol_spms - generate_thiol_spms.min()) / (
                    generate_thiol_spms.max() - generate_thiol_spms.min())
            generate_react_spms_std = np.concatenate([generate_cat_spms_std.reshape(1250, 20, 40, 1),
                                                      generate_imine_spms_std.reshape(1250, 20, 40, 1),
                                                      generate_thiol_spms_std.reshape(1250, 20, 40, 1)],
                                                     axis=3)

            generate_tag_scale = generate_tag.max() - generate_tag.min()  # 计算标签的尺度
            generate_tag_min = generate_tag.min()  # 记录标签的最小值
            generate_tag_std = (generate_tag - generate_tag_min) / generate_tag_scale  # 标签标准化

            # 加载数据
            cat_spms = np.load('Jishe_new_81/20X40/cat.npy')
            imine_spms = np.load('Jishe_new_81/20X40/imine.npy')
            thiol_spms = np.load('Jishe_new_81/20X40/thiol.npy')
            tag = np.load('./Reaction_Result/ddG.npy')

            print(cat_spms.shape, imine_spms.shape, thiol_spms.shape)  # 打印数据形状

            cat_spms = np.load('Jishe_new_81/20X40/cat.npy')
            imine_spms = np.load('Jishe_new_81/20X40/imine.npy')
            thiol_spms = np.load('Jishe_new_81/20X40/thiol.npy')
            tag = np.load('./Reaction_Result/ddG.npy')

            print(cat_spms.shape, imine_spms.shape, thiol_spms.shape)

            cat_spms_std = (cat_spms - cat_spms.min()) / (cat_spms.max() - cat_spms.min())
            imine_spms_std = (imine_spms - imine_spms.min()) / (imine_spms.max() - imine_spms.min())
            thiol_spms_std = (thiol_spms - thiol_spms.min()) / (thiol_spms.max() - thiol_spms.min())
            react_spms_std = np.concatenate([cat_spms_std.reshape(1075, 20, 40, 1),
                                             imine_spms_std.reshape(1075, 20, 40, 1), thiol_spms_std.reshape(1075, 20, 40, 1)],
                                            axis=3)

            react_spms_std_reshape = react_spms_std.reshape(1075, 20 * 40 * 3)
            react_spms_std_short = react_spms_std_reshape[:, np.where(react_spms_std_reshape.max(axis=0) != react_spms_std_reshape.min(axis=0))[0]]

            tag_scale = tag.max() - tag.min()
            tag_min = tag.min()
            tag_std = (tag - tag_min) / tag_scale

            generate_react_spms_std_shape = generate_react_spms_std.reshape(1250, 20 * 40 * 3)
            generate_react_spms_std_short = generate_react_spms_std_shape[:,
                                            np.where(react_spms_std_reshape.max(
                                                axis=0) != react_spms_std_reshape.min(axis=0))[0]]

            indices = np.arange(len(tag_std))  # 生成索引数组
            X_train, X_test, train_x_short, val_x_short, y_train, y_test, train_idx, test_idx = train_test_split(
                react_spms_std,
                react_spms_std_short,
                tag_std,
                indices,
                test_size=0.44,
                # test_size=0.3,
                random_state=fold + i)
            # 将索引存入json文件
            os.makedirs(args_opt.save_test_result, exist_ok=True)
            with open(os.path.join(args_opt.save_test_result, f'train_test_indices_{fold}_{i}.json'), 'a') as f:
                json.dump({'train_idx': train_idx.tolist(), 'test_idx': test_idx.tolist()}, f)

            y_train_short = y_train
            X_train = np.concatenate([X_train, generate_react_spms_std], axis=0)
            y_train = np.concatenate([y_train, generate_tag_std], axis=0)

            # X_train = generate_react_spms_std
            # y_train = generate_tag_std
            # train_x_short = generate_react_spms_std_short
            # y_train_short = y_train

            dataset = SPMSData
            dataset_train = dataset(X_train, y_train, partition='train', fold=i)
            dataset_valid = dataset(X_test, y_test, partition='val', fold=i)
            dataset_test = dataset(X_test, y_test, partition='test', fold=i)

            print(f'X_train: {X_train.shape}, X_val: {X_test.shape}, X_test: {X_test.shape}')

            shuffle_val = False
            shuffle_train = True
            shuffle_test = False

            train_loader = DataLoader(dataset_train, batch_size=args_opt.batch_size, shuffle=shuffle_train,
                                      num_workers=args_opt.num_workers, drop_last=True)
            valid_loader = DataLoader(dataset_valid, batch_size=args_opt.batch_size, shuffle=shuffle_val,
                                    num_workers=args_opt.num_workers, drop_last=False)
            test_loader = DataLoader(dataset_test, batch_size=args_opt.test_batch_size, shuffle=shuffle_test,
                                      num_workers=args_opt.num_workers, drop_last=False)  # 注意：测试的时候不能够drop_last=True，否则有可能测试数据不全

            data_loader = {'train': train_loader,
                           'val': valid_loader,
                           'test': test_loader}

            # create trainer
            trainer = Trainer(enc_module=enc_module,
                              data_loader=data_loader,
                              log=logger,
                              arg=args_opt,
                              best_step=0,
                              test_acc=-float('inf'),
                              tag_scale=tag_scale,
                              tag_min=tag_min)

            model_names = [
                           'CNN',

                           # 'DecisionTreeRegressor',
                           # 'SVR',
                           # 'BayesianRidge',
                           # 'LightGBM',

                           'RandomForestRegressor3',
                           # 'BaggingRegressor',
                           # "XGBoost",
                           # "CatBoost"
                           ]

            models = [
                enc_module,

                # DecisionTreeRegressor(criterion='absolute_error', max_depth=10),
                # SVR(kernel='rbf', C=1.0, epsilon=0.1),  # Support Vector Regressor
                # BayesianRidge(),  # Bayesian Regression
                # lightgbm.LGBMRegressor(n_estimators=500, learning_rate=0.1, num_leaves=31),  # LightGBM

                RandomForestRegressor(n_jobs=-1, criterion='squared_error',
                                      n_estimators=50, max_depth=10),
                # RandomForestRegressor(n_jobs=-1, criterion='mae'),
                # BaggingRegressor(),
                # xgb.XGBRegressor(),  # Add XGBoost model
                # CatBoostRegressor(iterations=500, learning_rate=0.1),  # Add CatBoost model
            ]

            if args_opt.mode == 'eval' or args_opt.resume == True:
                # CNN
                assert os.path.exists(os.path.join(args_opt.checkpoint_dir, 'model_best.pth.tar')), '指定模型文件未找到，请检查'
                # assert os.path.exists(os.path.join(args_opt.checkpoint_dir, 'checkpoint.pth.tar')), '指定模型文件未找到，请检查'
                logger.info('find a checkpoint, loading checkpoint from {}'.format(args_opt.checkpoint_dir))
                best_checkpoint = torch.load(os.path.join(args_opt.checkpoint_dir, 'model_best.pth.tar'), map_location=torch.device('cpu'))
                # best_checkpoint = torch.load(os.path.join(args_opt.checkpoint_dir, 'checkpoint.pth.tar'), map_location=torch.device('cpu'))

                logger.info('best model pack loaded')
                trainer.best_step = best_checkpoint['iteration']
                trainer.global_step = best_checkpoint['iteration']
                trainer.test_acc = best_checkpoint['test_acc']
                try:
                    trainer.enc_module.load_state_dict(best_checkpoint['enc_module_state_dict'])
                    # trainer.gnn_module.load_state_dict(best_checkpoint['gnn_module_state_dict'])
                except:
                    new_enc_module_state_dict = {f'module.{k}': v for k, v in best_checkpoint['enc_module_state_dict'].items()}
                    # new_gnn_module_state_dict = {f'module.{k}': v for k, v in best_checkpoint['gnn_module_state_dict'].items()}
                    trainer.enc_module.load_state_dict(new_enc_module_state_dict)
                    # trainer.gnn_module.load_state_dict(new_gnn_module_state_dict)
                trainer.optimizer.load_state_dict(best_checkpoint['optimizer'])
                logger.info('current best test R2 is: {}, at step: {}'.format(trainer.test_acc, trainer.best_step))

            elif args_opt.mode == 'train':
                # train all models
                for tmp_model, tmp_model_name in zip(models, model_names):
                    print(tmp_model_name + ':')
                    if tmp_model_name == 'CNN':
                        # print(None)
                        if not os.path.exists(args_opt.checkpoint_dir):
                            os.makedirs(args_opt.checkpoint_dir)
                            logger.info('no checkpoint for model: {}, make a new one at {}'.format(
                                args_opt.exp_name,
                                args_opt.checkpoint_dir))
                        logger.info('--------------start run{} training------------'.format(i))
                        trainer.train(args_opt.epoch, args_opt.stop_num)
                    else:
                        save_dir = os.path.join(args_opt.path_root, args_opt.exp_name, tmp_model_name)
                        # 使用os.makedirs()创建目录，如果目录不存在的话
                        os.makedirs(save_dir, exist_ok=True)

                        tmp_model.fit(train_x_short, y_train_short)
                        # Save the trained model
                        model_save_path = save_dir + f'/fold_{fold}_run_{i}.joblib'
                        joblib.dump(tmp_model, model_save_path)

                # test with all models
                logger.info('--------------start run{} testing------------'.format(i))
                model_predictions = []
                for tmp_model, tmp_model_name in zip(models, model_names):
                    print(tmp_model_name + ':')
                    if tmp_model_name == 'CNN':
                        assert os.path.exists(os.path.join(args_opt.checkpoint_dir, 'model_best.pth.tar')), '指定模型文件未找到，请检查'
                        # assert os.path.exists(os.path.join(args_opt.checkpoint_dir, 'checkpoint.pth.tar')), '指定模型文件未找到，请检查'
                        logger.info('find a checkpoint, loading checkpoint from {}'.format(args_opt.checkpoint_dir))
                        best_checkpoint = torch.load(os.path.join(args_opt.checkpoint_dir, 'model_best.pth.tar'), map_location=torch.device('cpu'))
                        # best_checkpoint = torch.load(os.path.join(args_opt.checkpoint_dir, 'checkpoint.pth.tar'),
                        #                              map_location=torch.device('cpu'))

                        logger.info('best model pack loaded')
                        trainer.best_step = best_checkpoint['iteration']
                        trainer.global_step = best_checkpoint['iteration']
                        trainer.test_acc = best_checkpoint['test_acc']
                        try:
                            trainer.enc_module.load_state_dict(best_checkpoint['enc_module_state_dict'])
                            # trainer.gnn_module.load_state_dict(best_checkpoint['gnn_module_state_dict'])
                        except:
                            new_enc_module_state_dict = {f'module.{k}': v for k, v in
                                                         best_checkpoint['enc_module_state_dict'].items()}
                            # new_gnn_module_state_.dict = {f'module.{k}': v for k, v in best_checkpoint['gnn_module_state_dict'].items()}
                            trainer.enc_module.load_state_dict(new_enc_module_state_dict)
                            # trainer.gnn_module.load_state_dict(new_gnn_module_state_dict)
                        trainer.optimizer.load_state_dict(best_checkpoint['optimizer'])
                        logger.info('current best test R2 is: {}, at step: {}'.format(trainer.test_acc, trainer.best_step))
                        result, pred_list, true_list = trainer.eval(partition='test', workbook=workbook, fold=i,
                                                        save_test_result=args_opt.save_test_result)
                        # print(y_test == true_list)
                        pred_list = np.array(pred_list)
                        model_predictions.append(pred_list)
                    else:
                        save_dir = os.path.join(args_opt.path_root, args_opt.exp_name, tmp_model_name)
                        model_save_path = save_dir + f'/fold_{fold}_run_{i}.joblib'
                        tmp_model = joblib.load(model_save_path)
                        pred_list = tmp_model.predict(val_x_short)
                        pred_list = np.array(pred_list)
                        model_predictions.append(pred_list)

                    predict_ddG = pred_list * tag_scale + tag_min

                    truth_ddG = y_test * tag_scale + tag_min
                    r2 = r2_score(truth_ddG, predict_ddG)
                    mae = mean_absolute_error(truth_ddG, predict_ddG)
                    # 计算MSE
                    mse = mean_squared_error(truth_ddG, predict_ddG)
                    # 计算RMSE
                    rmse = np.sqrt(mse)
                    print('DDG R2: %f, MAE: %f, RMSE: %f' % (r2, mae, rmse))

                # Combine CNN predictions and predictions from other models
                combined_predictions = np.mean(np.array(model_predictions), axis=0)

                # Convert predictions back to original scale
                predict_ddG = combined_predictions * tag_scale + tag_min

                truth_ddG = y_test * tag_scale + tag_min
                # total_predict_ddG.append(predict_ddG)
                # total_test_ddG.append(truth_ddG)

                r2 = r2_score(truth_ddG, predict_ddG)
                mae = mean_absolute_error(truth_ddG, predict_ddG)
                # 计算MSE
                mse = mean_squared_error(truth_ddG, predict_ddG)
                # 计算RMSE
                rmse = np.sqrt(mse)

                r2_list.append(r2)
                mae_list.append(mae)
                rmse_list.append(rmse)

                result_test.append([r2, mae, rmse])
                print('++++++++++++++++++++++')
                print('Ensemble DDG R2: %f, MAE: %f, RMSE: %f' % (r2, mae, rmse))
                print('++++++++++++++++++++++')

            else:
                print('select a mode')

                exit()


            if args_opt.mode == 'eval':
                # test with all models
                logger.info('--------------start run{} testing------------'.format(i))
                model_predictions = []
                for tmp_model, tmp_model_name in zip(models, model_names):
                    if tmp_model_name == 'CNN':
                        result, pred_list, true_list = trainer.eval(partition='test', workbook=workbook, fold=i,
                                                                    save_test_result=args_opt.save_test_result)
                        pred_list = np.array(pred_list)
                        model_predictions.append(pred_list)
                    else:
                        save_dir = os.path.join(args_opt.path_root, args_opt.exp_name, tmp_model_name)
                        model_save_path = save_dir + f'/fold_{fold}_run_{i}.joblib'
                        tmp_model = joblib.load(model_save_path)

                        start_time = time.time()
                        pred_list = tmp_model.predict(val_x_short)
                        end_time = time.time()
                        count = val_x_short.shape[0]
                        # 平均推理时间
                        average_inference_time = (end_time - start_time) / count
                        print(f"RF Average Inference Time: {average_inference_time * 1000:.3f} ms")

                        pred_list = np.array(pred_list)
                        model_predictions.append(pred_list)

                    predict_ddG = pred_list * tag_scale + tag_min

                    truth_ddG = y_test * tag_scale + tag_min
                    r2 = r2_score(truth_ddG, predict_ddG)
                    mae = mean_absolute_error(truth_ddG, predict_ddG)
                    # 计算MSE
                    mse = mean_squared_error(truth_ddG, predict_ddG)
                    # 计算RMSE
                    rmse = np.sqrt(mse)
                    print(tmp_model_name + ':')
                    print('DDG R2: %f, MAE: %f, RMSE: %f' % (r2, mae, rmse))

                # Combine CNN predictions and predictions from other models
                combined_predictions = np.mean(np.array(model_predictions), axis=0)

                # Convert predictions back to original scale
                predict_ddG = combined_predictions * tag_scale + tag_min

                truth_ddG = y_test * tag_scale + tag_min
                # print(truth_ddG == true_list)
                # total_predict_ddG.append(predict_ddG)
                # total_test_ddG.append(truth_ddG)

                r2 = r2_score(truth_ddG, predict_ddG)
                mae = mean_absolute_error(truth_ddG, predict_ddG)
                # 计算MSE
                mse = mean_squared_error(truth_ddG, predict_ddG)
                # 计算RMSE
                rmse = np.sqrt(mse)

                r2_list.append(r2)
                mae_list.append(mae)
                rmse_list.append(rmse)

                result_test.append([r2, mae, rmse])

                print('++++++++++++++++++++++')
                print('Ensemble DDG R2: %f, MAE: %f, RMSE: %f' % (r2, mae, rmse))
                print('++++++++++++++++++++++')

                # 测试结果的散点图展示
                if args_opt.plot_flag:
                    plt.scatter(truth_ddG, predict_ddG, color='blue', label='Predicted vs True', s=10, alpha=0.5)
                    plt.xlabel(r'Observed $\Delta\Delta$G (kcal mol$^{-1})$')
                    plt.ylabel(r'Predicted $\Delta\Delta$G (kcal mol$^{-1})$')
                    plt.plot([min(truth_ddG), max(truth_ddG)], [min(truth_ddG), max(truth_ddG)], color='red',
                             linestyle='--', linewidth=2,
                             label='Ideal')
                    plt.text(0.05, 0.9,
                             r'R$^2$' + '={:.4f}\n'.format(r2) + 'RMSE={:.4f}'.format(rmse),
                             color='black', fontsize=12,
                             ha='left', va='top',
                             transform=plt.gca().transAxes)
                    plt.gca().set_aspect('equal')
                    plt.savefig(os.path.join(args_opt.save_test_result, f'{fold}_{i}.png'), dpi=900, bbox_inches='tight')
                    plt.show()
                    plt.clf()

                # 将Tensor列表类型的predict_ddG和truth_ddG转换成numpy数组，并保存到一个csv文件中
                predict_ddG = np.array(predict_ddG).reshape(-1)
                # truth_ddG = np.array(truth_ddG).reshape(-1)
                # pT = pd.DataFrame({'predict_ddG': predict_ddG, 'truth_ddG': tag[test_idx].reshape(-1)})
                pT = pd.DataFrame({'predict_ddG': predict_ddG, 'truth_ddG': truth_ddG})
                pT.to_csv(os.path.join(args_opt.save_test_result, f'predict_truth_ddG_{fold}_{i}.csv'), index=True)

        r2_averge = np.average(np.array(r2_list))
        mae_averge = np.average(np.array(mae_list))
        rmse_averge = np.average(np.array(rmse_list))
        print('Ensemble DDG Average R2: %f, Average MAE: %f, Average RMSE: %f' % (r2_averge, mae_averge, rmse_averge))

        # 创建一个worksheet
        worksheet = workbook2.add_sheet("Fold" + str(fold))
        worksheet.write(0, 0, label="ID")
        worksheet.write(0, 1, label="R2")
        worksheet.write(0, 2, label="MAE")
        worksheet.write(0, 3, label="RMSE")

        for i in range(len(r2_list)):
            worksheet.write(i + 1, 0, label=i)
            worksheet.write(i + 1, 1, label=r2_list[i])
            worksheet.write(i + 1, 2, label=mae_list[i])
            worksheet.write(i + 1, 3, label=rmse_list[i])

        worksheet.write(len(r2_list) + 1, 0, label='Averge')
        worksheet.write(len(r2_list) + 1, 1, label=r2_averge)
        worksheet.write(len(r2_list) + 1, 2, label=mae_averge)
        worksheet.write(len(r2_list) + 1, 3, label=rmse_averge)

        # 保存
        # workbook2.save(os.path.join(args_opt.path_root, args_opt.exp_name) + '/prediction-cnn-rf-11.xls')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or eval')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--gpu_ids', type=list, default=[1], help='number of gpu')
    parser.add_argument('--folds', type=int, default=1)
    parser.add_argument('--cross_num', type=int, default=10)
    parser.add_argument('--plot_flag', type=bool, default=True)  # 是否绘制图表
    parser.add_argument('--learning_rate', type=int, default=1e-3)
    parser.add_argument('--weight_decay', type=int, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--stop_num', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=0)    # 这个项目中num_workers>0的时候dataloader会在某个提取数据的时候卡住
    parser.add_argument('--seed', type=int, default=222, help='random seed')

    # parser.add_argument('--path_root', type=str, default='/media/data1/Models_ly/3DChemical/denmark-2/train0.5-ensemble-spms-cnn-cat2-channel-20X40-trans-all/tain_70_again/',
    parser.add_argument('--path_root', type=str, default='/media/data1/Models_ly/3DChemical/denmark-2/train0.5-ensemble-spms-cnn-cat2-channel-20X40-trans-all/',
    # parser.add_argument('--path_root', type=str, default='/media/data1/Models_ly/3DChemical/denmark-2/train0.5-ensemble-spms-cnn-cat2-channel-20X40-trans-true/',
    # parser.add_argument('--path_root', type=str, default='/media/data1/Models_ly/3DChemical/denmark-2/train0.5-ensemble-spms-cnn-cat2-channel-20X40-trans-gen/',
    # parser.add_argument('--path_root', type=str, default='/media/data1/Models_ly/3DChemical/denmark-2/train0.5-ensemble-spms-cnn-20X40-trans-all/',
    # parser.add_argument('--path_root', type=str, default='/media/data1/Models_ly/3DChemical/denmark-2/train0.5-ensemble-spms-cnn-channel-20X40-trans-all/',
    # parser.add_argument('--path_root', type=str, default='/media/data1/Models_ly/3DChemical/denmark-2/train0.5-ensemble-spms-cnn-cat2-20X40-trans-all/',
                        help='path that checkpoint and logs will be saved and loaded. '
                             'It is assumed that the checkpoint file is placed under the directory ./checkpoints')
    parser.add_argument('--display_step', type=int, default=1, help='display training information in how many step')
    parser.add_argument('--log_step', type=int, default=1, help='log information in how many steps')
    parser.add_argument('--interval', type=int, default=1, help='log information in how many vals')

    args_opt = parser.parse_args()
    # train
    # main(args_opt)

    args_opt.mode = 'eval'
    # eval
    main(args_opt)
