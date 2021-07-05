import argparse
import yaml
import os

from config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train(case sensitive).")
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the model class to defense or attack(case sensitive).")
    parser.add_argument("--attack_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the attack class(case sensitive).")   

    parser.add_argument('--vGPU',
                        nargs='+',
                        type=int,
                        default=None,
                        help="Specify which GPUs to use.")
    parser.add_argument('--seed',
                        type=int,
                        default=2021,
                        help="random seed for initialization")


    args = parser.parse_args()

    ## Load Config File
    config = Config()

    if args.config:
        assert os.path.exists(args.config), "There's no '" + args.config + "' file."
        with open(args.config, "r") as load_f:
            config_parameter = yaml.load(load_f)
            config.load_parameter(config_parameter)

    ## Load Model Param File
    model_param_file = os.path.join('parameter', 'model_param', args.model_name.lower() + '.yaml')
    with open(model_param_file, 'r') as loaf_f:
        parameter = yaml.load(load_f)
        model_param_dict = parameter[args.task_name.lower()]
    
    ## Load Attack Param File
    attack_param_file = os.path.join('parameter', 'attack_param', args.attack_name.lower() + '.yaml')
    with open(attack_param_file, 'r') as loaf_f:
        parameter = yaml.load(load_f)
        parameter_tmp = parameter
        if parameter_tmp['model_difference']:
            parameter = parameter[args.model_name.lower()]
        if parameter_tmp['task_difference']:
            parameter = parameter[args.task_name.lower()]
        attack_param_dict = parameter


    



if __name__ == "__main__":
    main()