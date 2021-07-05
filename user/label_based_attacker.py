import os
import json
import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
from torch.utils.data import DataLoader

import model_loader
from config import *
import utils as util
import metric as module_metric
import data_loader as module_data_loader
import pickle
import datetime
import time


class LabelBasedAttacker(object):
    """[攻击者]

    Args:
        self.model ([]): [要攻击的模型]
        self.criterion ([]): 损失函数
        self.config ([]): 配置类
        self.attack_method ([]): 攻击方法
        self.use_gpu ([bool]): 是否使用GPU
        self.device_ids ([list]): 使用的GPU的id号
        self.attack_name ([str]): 攻击方法名称
        self.is_target ([bool]): 是否进行目标攻击
        self.target ([int]): 目标攻击的目标 
    """
    def __init__(self,model, metrics, criterion, config, attack_method):
        self.model = model
        self.criterion = criterion
        self.config = config
        self.attack_method = attack_method
        self.metrics = metrics


        #########################GPU配置
        self.use_gpu = False
        self.device_ids = [0]
        self.device = torch.device('cpu')
        if self.config.GPU['use_gpu']:
            if not torch.cuda.is_available():
                print("There's no GPU is available , Now Automatically converted to CPU device")
            else:
                message = "There's no GPU is available"
                self.device_ids = self.config.GPU['device_id']
                assert len(self.device_ids) > 0,message
                self.device = torch.device('cuda', self.device_ids[0])
                self.model = self.model.to(self.device)
                if len(self.device_ids) > 1:
                    self.model = nn.DataParallel(model, device_ids=self.device_ids)
                self.use_gpu = True


        #########################攻击信息
        self.attack_name = self.config.CONFIG['attack_name']
        #########################攻击方式---目标攻击设置
        self.is_target = False
        self.target = 0
        if 'is_target' in getattr(self.config,self.attack_name):
            self.is_target = getattr(self.config,self.attack_name)['is_target']
            self.target =  getattr(self.config,self.attack_name)['target']        

        # 各种路径配置
        #------------------------------------figure配置
        self.figure_dir = os.path.join('figure', self.config.CONFIG['model_name'] )



    def attack_batch(self, model, x, y):
        """[对一组数据进行攻击]]

        Args:
            x ([array]): [输入原始样本，]
            y (list, 无目标攻击中设置): [输入数据对应的标签]]. Defaults to [0].

        Returns:
            x_advs ([array]): [得到的对抗样本，四维]
            pertubations ([array]): [得到的对抗扰动，四维]
            nowLabels ([array]): [攻击后的样本标签，一维]
        """ 
        
        y = torch.Tensor(y).to(self.device).long()


        log = self.attack_method.attack(
            x, y, **getattr(self.config, self.config.CONFIG['attack_name']))

        return log


    def attack_one_img(self, x, y):   
        """[攻击一个图片]

        Args:
            x ([array]): [一个输入数据]
            y (list, 无目标攻击中设置): [输入数据对应的标签]]. Defaults to [0].

        Returns:
            x_adv ([array]): [得到的对抗样本，三维]
            pertubation ([array]): [得到的对抗扰动，三维]
            nowLabel ([int]): [攻击后的样本标签]
        """
        log = self.attack_method.attack(
            x, y, **getattr(self.config, self.config.CONFIG['attack_name']))

        return log


    def attack_set(self, model, data_loader):
        """[对一个数据集进行攻击]

        Args:
            data_loader ([DataLoader]): [数据加载器]

        Returns:
            acc [float]: [攻击后的准确率]
            mean [float]: 平均扰动大小
        """
        raise NotImplementedError



    def start_attack(self, data_loader):
        attack_method = self.config.Switch_Method['method']
        log = {}
        if attack_method == 'One_Sample_Attack':
            index = getattr(self.config, attack_method)['index']
            for i,(_, _, _, y, x) in enumerate(data_loader):
                if (i == index):
                    log = self.one_sample_attack(x[0], y[0])
        elif attack_method == 'Batch_Sample_Attack':
            log = self.batch_sample_attack(data_loader, **getattr(self.config, attack_method))
        elif attack_method == 'Exp_With_Pertub_Prop':
            # 对替换比例进行实验，参数查找
            log = self.exp_With_Pertub_Prop(data_loader, **getattr(self.config, attack_method))
        elif attack_method == 'Exp_With_Swap_In_Prop':
            # 对换回比例进行实验，参数查找
            log = self.exp_With_Swap_In_Prop(data_loader, **getattr(self.config, attack_method))

        return log



    def one_sample_attack(self, x, y):
        log = {}
        attack_log = self.attack_one_img(x,y)
        log['pre_data'] = x
        log['pre_label'] = y
        log.update(attack_log)
        return log

    def batch_sample_attack(self, data_loader, batch, json_name, json_save):
        """[攻击数目为batch的样本，还是一个个的攻击]

        Args:
            data_loader ([DataLoader]): [数据集加载器]
            batch ([int]): [要攻击的样本数目]
            json_name ([str]):  [存储中间数据的名称]
            json_save ([bool]): [是否存储json文件]

        Returns:
            log ([dict]):
                success_prop ([float]):             攻击成功的比例
                mean_pertub_prop ([float]):         平均扰动比例
                mean_replace_query_time ([float]):  平均访问次数
                mean_process_time ([float]):        平均攻击一个样本的时间
                swap_in_time ([int]):               换回的次数
                print_* ([str]):                    记录中间过程记录
            json_log ([json_file]):
                存在‘./tmp_parameters/batch_{}_result.json'.format(batch) 文件中
                success_prop ([float]):             攻击成功的比例
                mean_pertub_prop ([float]):         平均扰动比例
                mean_replace_query_time ([float]):  平均访问次数
                mean_process_time ([float]):        平均攻击一个样本的时间
                swap_in_time ([int]):               换回的次数
                pertub_prop_list ([list[float]]):   记录每个样本的扰动比例(只有攻击成功的)
                process_time_list ([list[int]]):    记录每一次的处理时间（只有攻击成功的）
                query_time_list ([list[int]]):      记录每一次攻击的访问次数（只有攻击成功的）
                success ([int]):                    攻击成功的个数
                sample_num ([int]):                 分类正确的个数
                
        """
        log = {}
        json_log = {}
        
        success = 0                 # 攻击成功的数目
        sample_num = 0              # 分类样本的数目
        mean_pertub_prop = 0        # 平均扰动比例
        mean_query_time = 0         # 平均访问次数
        mean_process_time = 0       # 平均处理时间
        swap_in_time = 0            # 换回的次数

        pertub_prop_list = []       # 记录每个样本的扰动比例(只有攻击成功的)
        process_time_list = []      # 记录每一次的处理时间（只有攻击成功的）
        query_time_list = []        # 记录每一次攻击的访问次数（只有攻击成功的）


        for i,(_, _, _, y, x) in enumerate(data_loader):
            if i == batch:
                break
            
            # 计算平均每个样本的处理时间
            starttime = datetime.datetime.now()
            one_log = self.one_sample_attack(x[0], y[0])
            endtime = datetime.datetime.now()
            process_time =  (endtime - starttime).seconds

            # 如果未正确分类，则跳过
            if not one_log['classification']:
                message = '第{}个样本未正确分类'.format(i)
                log['print_{}'.format(i)] = message
                print(message)
                continue

            sample_num += 1

            if(one_log['status']):
                # 攻击成功的情况
                success += 1
                mean_pertub_prop += one_log['swap_in_after_prop']
                replace_iter_num = one_log['replace_iter_num']
                replace_prop = one_log['swap_in_after_prop']
                query_time = one_log['replace_iter_num'] + one_log['swap_in_max_time']
                mean_query_time += query_time
                mean_process_time += process_time
                swap_in_time = one_log['swap_in_max_time']

                pertub_prop_list.append(replace_prop)
                process_time_list.append(process_time)
                query_time_list.append(query_time)

                message = '已完成第{}个样本，用时{}s, 攻击成功, 替换次数{}, 扰动比例{}'.format(i, process_time, replace_iter_num, replace_prop)
                print(message)
            else:
                # 攻击失败的情况
                message = '已完成第{}个样本，用时{}s, 攻击失败'.format(i, process_time)
                print(message)
            
            log['print_{}'.format(i)] = message
        
        message = '\n总共选择了{}各样本，{}个样本分类正确，{}个样本攻击成功，{}个样本攻击失败'.format(batch, sample_num, success, sample_num - success)
        print(message)
        log['print_last'] = message

        success_prop = success/sample_num
        mean_pertub_prop = mean_pertub_prop/success
        mean_query_time = mean_process_time/success

        log['success_prop'] = success_prop
        log['mean_pertub_prop'] = mean_pertub_prop
        log['mean_replace_query_time'] = mean_query_time
        log['mean_process_time'] = mean_process_time
        log['swap_in_time'] = swap_in_time

        for key, value in log.items():
            if 'print' not in key:
                json_log[key] = value


        json_log['success'] = success
        json_log['sample_num'] = sample_num
        json_log['pertub_prop_list'] = pertub_prop_list
        json_log['process_time_list'] = process_time_list
        json_log['query_time_list'] = query_time_list
        
        if json_save:
            with open(json_name.format('batch_sample_attack',batch, time.strftime("%m-%d_%H-%M")), 'w') as fw:
                json.dump(json_log,fw)

        return log

    def exp_With_Pertub_Prop(self, data_loader, batch, json_name, json_save, pertub_prop_list, figure_dir):
        """[对扰动比例进行实验]

        Args:
            data_loader ([DataLoader]): [数据集加载器]
            batch ([int]): [要攻击的样本数目]
            json_name ([str]):  [存储中间数据的名称]
            json_save ([bool]): [是否存储json文件]
            pertub_prop_list ([list[list]]):    [各种替换比例，对其进行研究]
            figure_name (str):  [存储图片的文件夹]

        Returns:
            log ([dict]):
                success_prop ([float]):             攻击成功的比例
                mean_pertub_prop ([float]):         平均扰动比例
                mean_replace_query_time ([float]):  平均访问次数
                mean_process_time ([float]):        平均攻击一个样本的时间
                swap_in_time ([int]):               换回的次数
                print_* ([str]):                    记录中间过程记录
            json_log ([dict]):
                success_prop_list ([list[float]]):              各个扰动比例对应的成功率
                mean_pertub_prop_list ([list[float]]):          各个扰动比例对应的扰动比例
                mean_replace_query_time_list ([list[float]]):   各个扰动比例对应的访问时间
                mean_process_time_list ([list[float]]):         各个扰动比例对应的处理时间
                pertub_prop_list ([list[list]]):               各个扰动比例
        """
        log = {}
        json_log = {}


        success_prop_list = []
        mean_pertub_prop_list = []
        mean_replace_query_time_list = []
        mean_process_time_list = []
        
        for pertub_prop in pertub_prop_list:
            self.config.Synonym_Replace['pertub_prop'] = pertub_prop
            print('now pertub_prop = {}, Start Attack----------------------------------------'.format(pertub_prop))
            exp_log = self.batch_sample_attack(data_loader, batch, json_name, json_save)

            success_prop_list.append(exp_log['success_prop'])
            mean_pertub_prop_list.append(exp_log['mean_pertub_prop'])
            mean_replace_query_time_list.append(exp_log['mean_replace_query_time'])
            mean_process_time_list.append(exp_log['mean_process_time'])

            for key, value in exp_log.items():
                log['pertub_prop = ' + str(pertub_prop)+'___'+ key] = value

        # 存储中间结果到json文件中
        json_log['success_prop_list'] = success_prop_list
        json_log['mean_pertub_prop_list'] = mean_pertub_prop_list
        json_log['mean_replace_query_time_list'] = mean_replace_query_time_list
        json_log['mean_process_time_list'] = mean_process_time_list
        json_log['pertub_prop_list'] = pertub_prop_list

        time_str = time.strftime("%m-%d_%H-%M")
        with open(json_name.format('Exp_With_Pertub_Prop',batch, time_str), 'w') as fw:
                json.dump(json_log,fw)

        # 将中间结果形式化为图片，存放到figure_name对应的文件夹中
        figure_dir_name = figure_dir.format(batch, time_str)
        util.ensure_dir(figure_dir_name)
        pertub_prop_str_list = [str(prop) for prop in pertub_prop_list]
        # 替换比例--攻击成功率
        self.plot_line_figure(pertub_prop_str_list, success_prop_list, "pertub_prop", "success_prop", "pertub_prop  vs  success_prop", figure_dir_name)
        # 替换比例--扰动比例
        self.plot_line_figure(pertub_prop_str_list, mean_pertub_prop_list, "pertub_prop", "mean_pertub_prop", "pertub_prop  vs  mean_pertub_prop", figure_dir_name)
        # 替换比例--访问次数
        self.plot_line_figure(pertub_prop_str_list, mean_replace_query_time_list, "pertub_prop", "mean_replace_query_time", "pertub_prop  vs  mean_replace_query_time", figure_dir_name)
        # 替换比例--处理时间
        self.plot_line_figure(pertub_prop_str_list, mean_process_time_list, "pertub_prop", "mean_process_time", "pertub_prop  vs  mean_process_time", figure_dir_name)
        
        return log

    def exp_With_Swap_In_Prop(self, data_loader, batch, json_name, json_save, swap_in_prop_list, figure_dir):
        """[对换回比例进行实验]

        Args:
            data_loader ([DataLoader]): [数据集加载器]
            batch ([int]): [要攻击的样本数目]
            json_name ([str]):  [存储中间数据的名称]
            json_save ([bool]): [是否存储json文件]
            swap_in_prop_list ([list[list]]):    [各种换回比例，对其进行研究]
            figure_name (str):  [存储图片的文件夹]

        Returns:
            log ([dict]):
                success_prop ([float]):             攻击成功的比例
                mean_pertub_prop ([float]):         平均扰动比例
                mean_replace_query_time ([float]):  平均访问次数
                mean_process_time ([float]):        平均攻击一个样本的时间
                swap_in_time ([int]):               换回的次数
                print_* ([str]):                    记录中间过程记录
            json_log ([dict]):
                success_prop_list ([list[float]]):              各个扰动比例对应的成功率
                mean_pertub_prop_list ([list[float]]):          各个扰动比例对应的扰动比例
                mean_replace_query_time_list ([list[float]]):   各个扰动比例对应的访问时间
                mean_process_time_list ([list[float]]):         各个扰动比例对应的处理时间
                pertub_prop_list ([list[list]]):               各个扰动比例
        """
        log = {}
        json_log = {}


        success_prop_list = []
        mean_pertub_prop_list = []
        mean_replace_query_time_list = []
        mean_process_time_list = []
        
        for swap_in_prop in swap_in_prop_list:
            self.config.Synonym_Replace['swap_in_prop'] = swap_in_prop
            print('now swap_in_prop = {}, Start Attack----------------------------------------'.format(swap_in_prop))
            exp_log = self.batch_sample_attack(data_loader, batch, json_name, json_save)

            success_prop_list.append(exp_log['success_prop'])
            mean_pertub_prop_list.append(exp_log['mean_pertub_prop'])
            mean_replace_query_time_list.append(exp_log['mean_replace_query_time'])
            mean_process_time_list.append(exp_log['mean_process_time'])

            for key, value in exp_log.items():
                log['swap_in_prop = ' + str(swap_in_prop)+'___'+ key] = value

        # 存储中间结果到json文件中
        json_log['success_prop_list'] = success_prop_list
        json_log['mean_pertub_prop_list'] = mean_pertub_prop_list
        json_log['mean_replace_query_time_list'] = mean_replace_query_time_list
        json_log['mean_process_time_list'] = mean_process_time_list
        json_log['swap_in_prop_list'] = swap_in_prop_list

        time_str = time.strftime("%m-%d_%H-%M")
        with open(json_name.format('Exp_With_Swap_In_Prop',batch, time_str), 'w') as fw:
                json.dump(json_log,fw)

        # 将中间结果形式化为图片，存放到figure_name对应的文件夹中
        figure_dir_name = figure_dir.format(batch, time_str)
        util.ensure_dir(figure_dir_name)
        swap_in_prop_list = [str(prop) for prop in swap_in_prop_list]
        # 替换比例--攻击成功率
        self.plot_line_figure(swap_in_prop_list, success_prop_list, "swap_in_prop", "success_prop", "swap_in_prop  vs  success_prop", figure_dir_name)
        # 替换比例--扰动比例
        self.plot_line_figure(swap_in_prop_list, mean_pertub_prop_list, "swap_in_prop", "mean_pertub_prop", "swap_in_prop  vs  mean_pertub_prop", figure_dir_name)
        # 替换比例--访问次数
        self.plot_line_figure(swap_in_prop_list, mean_replace_query_time_list, "swap_in_prop", "mean_replace_query_time", "swap_in_prop  vs  mean_replace_query_time", figure_dir_name)
        # 替换比例--处理时间
        self.plot_line_figure(swap_in_prop_list, mean_process_time_list, "swap_in_prop", "mean_process_time", "swap_in_prop  vs  mean_process_time", figure_dir_name)
        
        return log

        

    def plot_line_figure(self, x_axis, y_axis, xlabel, ylabel, title, save_dir):
        plt.figure()
        plt.plot(x_axis,y_axis, c = "r", label = "red")
        #美化图的操作
        plt.xticks(rotation = 45)  #使x轴的数字旋转45°
        plt.xlabel(xlabel) #x轴标签
        plt.ylabel(ylabel) #y轴标签
        plt.title(title) #此图像的标题
        plt.legend(loc = "best") #为图像生成legend，loc参数为best指，在最适合的地方显示
        plt.savefig(os.path.join(save_dir, title + '.jpg'))
  

    def _eval_metrics(self,logits,targets):
        """[多种metric的运算]

        Args:
            logits ([array]): [网络模型输出]
            targets ([array]): [标签值]

        Returns:
            acc_metrics [array]: [多个metric对应的结果]
        """
        acc_metrics = np.zeros(len(self.metrics))
        for i,metric in enumerate(self.metrics):
            acc_metrics[i] = metric(logits,targets)
        return acc_metrics
        