import os
import argparse
from data_loader import DataLoader
from trainer import Trainer
from evaluator import Evaluator
from util import Util


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Argument Parser for IntroVAE')
    parser.add_argument("--image_dir", dest="image_dir", type=str, default='')
    parser.add_argument("--latent_dimension", dest="latent_dimension", type=int, default=20)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=64)
    parser.add_argument("--lr", dest="lr", type=float, default=0.001)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=50)
    parser.add_argument("--margin", dest="margin", type=int, default=0.2)
    parser.add_argument("--clip_value", dest="clip_value", type=float, default=5)
    parser.add_argument("--wdecay", dest="wdecay", type=float, default=0.00001)
    parser.add_argument("--validate_every", dest="validate_every", type=int, default=5)
    parser.add_argument("--mode", dest="mode", type=int, default=0)
    parser.add_argument("--cuda", dest="cuda", type=bool, default=False)

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    parser.add_argument("--model_dir", dest="model_dir", type=str, default=model_dir)
    parser.add_argument("--model_file_name", dest="model_file_name", type=str, default="model_weights_5.t7")
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    parser.add_argument("--result_dir", dest="result_dir", type=str, default=result_dir)

    return parser.parse_args()


def main():
    params = parse_arguments()
    print("Constructing data loaders...")
    util = Util(params)
    dl = DataLoader(params, util)
    evaluator = Evaluator(params, dl, util)
    print("Constructing data loaders...[OK]")

    if params.mode == 0:
        print("Training...")
        t = Trainer(params, dl, evaluator, util)
        t.train()
        print("Training...[OK]")


if __name__ == '__main__':
    main()