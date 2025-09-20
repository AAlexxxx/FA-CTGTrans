import os
import argparse
import warnings
from trainer import trainer
from test import tester
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


parser = argparse.ArgumentParser()

# ========  Experiments Name ================
parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
parser.add_argument('--experiment_description', default='Exp1',   type=str, help='experiment name')
parser.add_argument('--run_description',        default='run1',     type=str, help='run name')
# ========= Select the DATASET ==============
parser.add_argument('--dataset',                default='CTU_CHB',           type=str, help='CTU-CHB, HUAYI')
parser.add_argument('--seed_id',                default='56',             type=str, help='to fix a seed while training')
# ========= Experiment settings ===============
parser.add_argument('--data_path',              default=r'ctg',           type=str,   help='Path containing dataset')
parser.add_argument('--num_runs',               default=3,                 type=int,   help='Number of consecutive run with different seeds')
parser.add_argument('--device',                 default='cuda',            type=str,   help='cpu or cuda')

args = parser.parse_args()

if __name__ == "__main__":
    # TRAIN
    trainer = trainer(args)
    trainer.train()

    # TEST
    tester = tester(args)
    tester.test()
