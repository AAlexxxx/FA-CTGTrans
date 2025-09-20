import torch
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
import os
import collections
import numpy as np
import warnings
import sklearn.exceptions

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from model import FA_CTGTrans
from dataloader import data_generator
from configs.data_configs import get_dataset_class
from configs.hparams import get_hparams_class
from utils import AverageMeter, to_device,_save_metrics_test, copy_Files
from utils import fix_randomness, starting_logs, _calc_metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, \
    roc_curve,auc, brier_score_loss,accuracy_score,average_precision_score
import datetime


class tester(object):
    def __init__(self, args):
        # dataset parameters
        self.dataset = args.dataset
        self.seed_id = args.seed_id

        self.device = torch.device(args.device)

        # Exp Description
        self.run_description = args.run_description
        self.experiment_description = args.experiment_description

        # paths
        self.home_path = os.getcwd()
        self.save_dir = os.path.join(os.getcwd(), "experiments_logs_NewData")
        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, self.run_description)
        os.makedirs(self.exp_log_dir, exist_ok=True)
        self.data_path = args.data_path

        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()
        self.hparams = self.hparams_class.train_params

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class("supervised")
        return dataset_class(), hparams_class()

    def load_data(self, data_type):
        self.train_dl, self.val_dl, self.test_dl, self.cw_dict = \
            data_generator(self.data_path, data_type, self.hparams)

    def calc_results_per_run(self):
        acc_list = []
        f1_list = []

        acc, f1 = _calc_metrics(self.pred_labels, self.true_labels, self.dataset_configs.class_names)

        acc_list.append(acc)
        f1_list.append(f1)

        acc_mean = np.mean(acc_list) * 100
        acc_std = np.std(acc_list) * 100
        f1_mean = np.mean(f1_list) * 100
        f1_std = np.std(f1_list) * 100

        self.logger.debug(f'【Accuracy Mean: {acc_mean:.2f}% | Std: {acc_std:.2f}%】')
        self.logger.debug(f'【F1-Score Mean: {f1_mean:.2f}% | Std: {f1_std:.2f}%】')
        return (acc_mean, acc_std), (f1_mean, f1_std)

    def test(self):
        copy_Files(self.exp_log_dir)  # save a copy of training files

        self.metrics = {'accuracy': [], 'f1_score': []}

        # fixing random seed
        fix_randomness(int(self.seed_id))

        self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.exp_log_dir, self.seed_id)
        self.logger.debug(self.hparams)

        self.load_data(self.dataset)

        model = FA_CTGTrans(configs=self.dataset_configs, hparams=self.hparams)
        model.to(self.device)

        loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
            betas=(0.9, 0.99)
        )

        weight_tensor = torch.tensor(list(self.cw_dict.values()), dtype=torch.float32).to(self.device)
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=weight_tensor)

        # TESTING
        print(" === Evaluating on TEST set ===")
        self.evaluate(model, self.test_dl)

        _save_metrics_test(self.pred_labels, self.true_labels, self.exp_log_dir, self.home_path,
                      self.dataset_configs.class_names)

    def evaluate(self, model, dataset):
        model.to(self.device).eval()

        save_path = os.path.join(self.exp_log_dir, 'validation_best')
        model.load_state_dict(torch.load(save_path, map_location='cpu'))

        total_loss_ = []

        self.pred_labels = np.array([])
        self.true_labels = np.array([])
        model_probs = []

        sensitivity_list = []
        specificity_list = []
        quality_index_list = []
        auc_list = []
        brier_list = []
        acc_list = []
        f1_list = []
        mcc_list = []
        acc_Conf_list = []

        with torch.no_grad():
            for batches in dataset:
                batches = to_device(batches, self.device)
                data = batches['samples'].float()
                labels = batches['labels'].long()

                # forward pass
                predictions = model(data)
                probs = torch.softmax(predictions, dim=1)[:, 1]
                model_probs.extend(probs.cpu().numpy())

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss_.append(loss.item())
                pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability

                self.pred_labels = np.append(self.pred_labels, pred.cpu().numpy())
                self.true_labels = np.append(self.true_labels, labels.data.cpu().numpy())

                self.trg_loss = torch.tensor(total_loss_).mean()  # average loss

                sensitivity, specificity, quality_index, AUC = calculate_metrics(
                    self.true_labels, self.pred_labels)
                brier = brier_score_loss(self.true_labels, self.pred_labels)
                mcc = matthews_corrcoef(self.true_labels, self.pred_labels)

                true_labels = np.array(self.true_labels)
                model1_probs = np.array(model_probs)
                fpr, tpr, _ = roc_curve(true_labels, model1_probs)
                roc_auc = auc(fpr, tpr)

                sensitivity_list.append(sensitivity)
                specificity_list.append(specificity)
                quality_index_list.append(quality_index)
                auc_list.append(roc_auc)
                brier_list.append(brier)
                mcc_list.append(mcc)

                cm = confusion_matrix(self.true_labels, self.pred_labels)

                TP = cm[1, 1]
                FP = cm[0, 1]
                FN = cm[1, 0]
                TN = cm[0, 0]

                acc = accuracy_score(self.true_labels, self.pred_labels)
                acc_Conf = (TP + TN) / (TP+TN+FP+FN)
                f1 = f1_score(self.true_labels, self.pred_labels, average='weighted')

                acc_list.append(acc)
                acc_Conf_list.append(acc_Conf)
                f1_list.append(f1)

        plot_confusion_matrix(self.true_labels, self.pred_labels, self.dataset_configs.class_names, self.exp_log_dir)

        plot_precision_recall_curve(self.true_labels, self.pred_labels, self.exp_log_dir)

        sensitivity, specificity, quality_index, AUC = calculate_metrics(
            self.true_labels, self.pred_labels)
        self.logger.debug(f'Val_Sen: {sensitivity:.4f}')
        self.logger.debug(f'Val_Sen: {specificity:.4f}')
        self.logger.debug(f'Val_Sen: {quality_index:.4f}')
        self.logger.debug(f'----------------------------------------------------------------------------------')
        self.logger.debug(f'self.pred_labels List: {self.pred_labels}')
        self.logger.debug(f'self.true_labels List: {self.true_labels}')
        self.logger.debug(f'----------------------------------------------------------------------------------')


        self.logger.debug(f'----------------------------------------------------------------------------------')
        self.logger.debug(f'Sensitivity List: {sensitivity_list}')
        self.logger.debug(f'Specificity List: {specificity_list}')
        self.logger.debug(f'Quality Index List: {quality_index_list}')
        self.logger.debug(f'AUC List: {auc_list}')
        self.logger.debug(f'MCC List: {mcc_list}')
        self.logger.debug(f'Brier Score List: {brier_list}')
        self.logger.debug(f'Accuracy List: {acc_list}')
        self.logger.debug(f'Acc_Conf List: {acc_Conf_list}')
        self.logger.debug(f'F1 Score List: {f1_list}')
        self.logger.debug(f'----------------------------------------------------------------------------------')

        sensitivity_mean = np.mean(sensitivity_list) * 100
        sensitivity_std = np.std(sensitivity_list) * 100

        specificity_mean = np.mean(specificity_list) * 100
        specificity_std = np.std(specificity_list) * 100

        quality_index_mean = np.mean(quality_index_list) * 100
        quality_index_std = np.std(quality_index_list) * 100

        auc_mean = np.mean(auc_list)
        auc_std = np.std(auc_list)

        mcc_mean = np.mean(mcc_list)
        mcc_std = np.std(mcc_list)

        brier_mean = np.mean(brier_list)
        brier_std = np.std(brier_list)


        acc_mean = np.mean(acc_list) * 100
        acc_std = np.std(acc_list) * 100

        ACC_Conf_mean = np.mean(acc_Conf_list) * 100
        ACC_Conf_std = np.std(acc_Conf_list) * 100

        f1_mean = np.mean(f1_list) * 100
        f1_std = np.std(f1_list) * 100

        self.logger.debug(f'----------------------------------------------------------------------------------')
        self.logger.debug(f'Mean Accuracy: {acc_mean:.2f}% (Std: {acc_std:.2f}%)')
        self.logger.debug(f'Mean ACC_Conf: {ACC_Conf_mean:.2f}% (Std: {ACC_Conf_std:.2f}%)')
        self.logger.debug(f'Mean Sensitivity: {sensitivity_mean:.2f}% (Std: {sensitivity_std:.2f}%)')
        self.logger.debug(f'Mean Specificity: {specificity_mean:.2f}% (Std: {specificity_std:.2f}%)')
        self.logger.debug(f'【Mean Quality Index】: {quality_index_mean:.2f}% (Std: {quality_index_std:.2f}%)')
        self.logger.debug(f'Mean F1-Score: {f1_mean:.2f}% (Std: {f1_std:.2f}%)')
        self.logger.debug(f'Mean AUC: {auc_mean:.4f} (Std: {auc_std:.4f})')
        self.logger.debug(f'Mean MCC: {mcc_mean:.4f} (Std: {mcc_std:.4f})')
        self.logger.debug(f'Mean Brier Score: {brier_mean:.4f} (Std: {brier_std:.4f})')
        self.logger.debug(f'----------------------------------------------------------------------------------')

        return {
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "ACC_Conf_mean": ACC_Conf_mean,
            "ACC_Conf_std": ACC_Conf_std,
            "sensitivity_mean": sensitivity_mean,
            "sensitivity_std": sensitivity_std,
            "specificity_mean": specificity_mean,
            "specificity_std": specificity_std,
            "quality_index_mean": quality_index_mean,
            "quality_index_std": quality_index_std,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "mcc_mean": mcc_mean,
            "mcc_std": mcc_std,
            "brier_mean": brier_mean,
            "brier_std": brier_std
        }



def plot_confusion_matrix(true_labels, pred_labels, class_names, save_dir):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f'confusion_matrix_test_{current_time}.png'

    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, file_name))
    plt.show()


def plot_precision_recall_curve(true_labels, pred_probs, save_dir):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f'precision_recall_curve_{current_time}.png'

    precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
    ap = average_precision_score(true_labels, pred_probs)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, marker='.',color='b', lw=2)

    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.title(f'Precision–Recall Curve (AP = {ap:.4f})', fontsize=20)
    plt.xticks(np.arange(0, 1.1, 0.2), fontsize=16)
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=16)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.legend(loc='lower left', fontsize=12, frameon=False)

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, file_name),dpi=300, bbox_inches='tight')


def calculate_metrics(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)

    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TN = cm[0, 0]

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    quality_index = (sensitivity * specificity) ** 0.5

    fpr, tpr, thresholds = roc_curve(true_labels,pred_labels)
    AUC = auc(fpr, tpr)

    return sensitivity, specificity, quality_index, AUC




