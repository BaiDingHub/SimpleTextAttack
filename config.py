import time
class Config(object):
    def __init__(self):
        self.ENV = 'default'            #当前的环境参数
        self.Introduce = 'Not at the moment'    #对此次实验的描述
        self.VERSION = 1                #当前的版本


        ##################################################GPU配置
        self.GPU = dict(
            use_gpu = True,             #是否使用GPU，True表示使用
            device_id = [0],            #所使用的GPU设备号，type=list
        )


        self.CONFIG = dict(
            dataset_name = 'yahoo',     # 数据集
            model_name = 'Bert',       #攻击模型的名称
            metrics = ['accuracy'],        # 评价标准的名称（metric文件夹中）
            attack_name = 'Random_Replace_Optim_Info_V2',       #设定攻击方法的名称
        )



        #################################################模型选择
        ##########################模型参数
        self.Bert = dict(
            pretrained_dir = '/home/yuzhen/nlp/pretrained/bert/' + self.CONFIG['dataset_name'],                # 预训练模型所在的位置
            nclasses = 10,                      # 所对应的数据集的类别数目
            max_seq_length = 256,                # 文本序列最大长度
            batch_size = 32,                    # 分类时的batch数目
        )
        self.WordCNN = dict(
            embedding_path = '/home/yuzhen/nlp/embedding/glove/glove.6B.200d.txt',  # 词向量所在的位置
            nclasses = 4,                        # 所对应的数据集的类别数目
            batch_size = 32,                    # 分类时的batch数目
            target_model_path = '/home/yuzhen/nlp/pretrained/WordCNN/' + self.CONFIG['dataset_name'],               # 预训练模型的位置
        )

        self.WordLSTM = dict(
            embedding_path = '/home/yuzhen/nlp/embedding/glove/glove.6B.200d.txt',  # 词向量所在的位置
            nclasses = 10,                        # 所对应的数据集的类别数目
            batch_size = 32,                    # 分类时的batch数目
            target_model_path = '/home/yuzhen/nlp/pretrained/WordLSTM/' + self.CONFIG['dataset_name'],               # 预训练模型的位置
        )
        self.Bert_NLI = dict(
            pretrained_dir = '/home/yuzhen/nlp/pretrained/bert/' + self.CONFIG['dataset_name'],                # 预训练模型所在的位置
            max_seq_length = 128,                # 文本序列最大长度
            batch_size = 32,                    # 分类时的batch数目
        )
        self.ESIM = dict(
            pretrained_file = '/home/yuzhen/nlp/pretrained/esim/best.pth.tar',
            worddict_path = '/home/yuzhen/nlp/embedding/glove/glove.6B.200d.txt',
            batch_size = 32,
        )


        #################################################损失函数选择



        ## 数据集
        #### 数据集参数
        self.IMDB = dict(
            dirname = '/home/yuzhen/research/Experiment_label_based/textClassifyDataset/IMDB',            #外卖情感分类数据集存放的文件夹
            prop = 0.5,         #训练集所占的比例
            model_name = 'bert-base-uncased',       #模型采用的分词策略
            len_seq = 512,                  #一个序列的长度
        )
        self.AdvDataset = dict(
            dataset_path = '/home/yuzhen/nlp/adv_dataset/' + self.CONFIG['dataset_name'],       # 数据集
        )



        ## 攻击方法
        self.Synonym_Replace = dict(
            is_target = False,          # 控制攻击方式，目标攻击、无目标攻击
            target = 3,                 # 目标攻击的目标
            len_seq = 512,              # 输入样本的最大长度
            pertub_prop = [1, 2, 0.05, 0.1, 0.2],          # 每次的同义词替换比例
            swap_max_time = 700,        # 换回的次数
            swap_in_prop = [1, 2, 3],   # 每次换回的word个数
            dirname = './tmp_parameters', # 中间参数的存储位置
        )
        self.Random_Replace = dict(
            is_target = False,          # 控制攻击方式，目标攻击、无目标攻击
            target = 3,                 # 目标攻击的目标
            pertub_prop = [1, 2, 0.05, 0.1, 0.2],          # 每次的同义词替换比例或个数
            replace_max_time = 20000,  # 替换的最大次数
            swap_max_time = 700,        # 换回的次数
            swap_in_prop = [1, 2, 3],   # 每次换回的word个数
            dirname = './tmp_parameters', # 中间参数的存储位置
            synonym_pick_way = 'embedding',     # 同义词挑选方式，可选embedding, nltk
            synonym_num = 4,                  # 选择的同义词的数目
            embedding_path = '/home/yuzhen/nlp/synonym/counter-fitted-vectors.txt',                # 当synonym_pick_way 选择embedding 时，使用该参数
            cos_path = '/home/yuzhen/nlp/synonym/counter-fitting-cos-sim.txt',                      # 当synonym_pick_way 选择embedding 时，使用该参数
        )
        self.Random_Replace_Optim_NLI = dict(
            is_target = False,          # 控制攻击方式，目标攻击、无目标攻击
            target = 3,                 # 目标攻击的目标
            pertub_prop = [1, 2, 0.2],          # 每次的同义词替换比例或个数
            replace_max_time = 5000,  # 替换的最大次数
            swap_max_time = 10,        # 换回的次数
            swap_in_prop = [1, 2, 3],   # 每次换回的word个数
            dirname = './tmp_parameters', # 中间参数的存储位置
            synonym_pick_way = 'embedding',     # 同义词挑选方式，可选embedding, nltk
            synonym_num = 4,                  # 选择的同义词的数目
            embedding_path = '/home/yuzhen/nlp/synonym/counter-fitted-vectors.txt',                # 当synonym_pick_way 选择embedding 时，使用该参数
            cos_path = '/home/yuzhen/nlp/synonym/counter-fitting-cos-sim.txt',                      # 当synonym_pick_way 选择embedding 时，使用该参数
            population_num = 4,             # 替换操作生成的population的数目
            mutate_num = 6,                 # 每次迭代中换回的次数
            top_k = 4,                      # 每次迭代后保存的样本的数目
        )


        #################################################log
        self.Checkpoint = dict(
            log_dir = './log',          #log所在的文件夹
            log_filename = '{}_{}_V{}_{}.log'.format(self.CONFIG['model_name'], self.CONFIG['attack_name'],
                self.VERSION, time.strftime("%m-%d_%H-%M")),              #log文件名称
        )
        ## 针对attacker的特定函数
        self.Switch_Method = dict(
            method = 'Batch_Sample_Attack',        # 可选['One_Sample_Attack', 'Batch_Sample_Attack'，'Exp_With_Pertub_Prop','Exp_With_Swap_In_Prop']
        )
        self.Batch_Sample_Attack = dict(
            batch = 1000,                       # 攻击的样本数目
            json_name = './tmp_parameters/' + self.CONFIG['dataset_name'] +'_' + self.CONFIG['attack_name'] + '_' + 'batch_{}_result_{}.json',      # 某些实验结果的输出.format(batch, time)
            json_save = True,                 # 是否存储json文件
        )
        self.One_Sample_Attack = dict(
            index = 2,                      # 选择攻击的第index样本
        )

    def log_output(self):
        log = {}
        log['ENV'] = self.ENV
        log['Introduce'] = self.Introduce
        log['CONFIG'] = self.CONFIG
        for name,value in self.CONFIG.items():
            if type(value) is str and hasattr(self,value):
                log[value] = getattr(self,value)
            else:
                log[name] = value
        log['Switch_Method'] = self.Switch_Method['method']
        log['Switch_Method_Value'] = getattr(self, self.Switch_Method['method'])

        return log

    
    def load_parameter(self, parameter):
        """[加载parameter内的参数]

        Args:
            parameter ([dict]): [由yaml文件导出的字典，包含有各个属性]
        """
        for key, value in parameter.items():
            if hasattr(self, key):
                if type(value) is dict:
                    orig_config = getattr(self, key)
                    if orig_config.keys() == value.keys():
                        setattr(self, key, value)
                    else:
                        redundant_key = value.keys() - orig_config.keys()
                        if redundant_key:
                            msg = "there are many redundant keys in config file, e.g.:  " + str(redundant_key)
                            raise msg
                        
                        lack_key = orig_config.keys() - value.keys()
                        if lack_key:
                            msg = "there are many lack keys in config file, e.g.:  " + str(lack_key)
                            raise msg
                else:
                    setattr(self, key, value)
            else:
                setattr(self, key, value)
        
    

    
    