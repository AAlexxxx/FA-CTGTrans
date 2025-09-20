import torch
import torch.nn.functional as F
import datetime
import os
import collections
import numpy as np

import warnings
import sklearn.exceptions
import seaborn as sns

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import confusion_matrix, roc_auc_score
from model import FA_CTGTrans
from dataloader import data_generator
# from dataloader import get_class_weight
from configs.data_configs import get_dataset_class
from configs.hparams import get_hparams_class
from utils import AverageMeter, to_device, _save_metrics_val, copy_Files
from utils import fix_randomness, starting_logs, _calc_metrics
import matplotlib.pyplot as plt


class trainer(object):
    def __init__(self, args):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
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

        # Specify hparams
        self.hparams = self.hparams_class.train_params

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class("supervised")
        return dataset_class(), hparams_class()

    def load_data(self, data_type):
        self.train_dl, self.val_dl, self.test_dl, self.cw_dict = \
            data_generator(self.data_path, data_type, self.hparams)

    def calc_results_per_run(self):
        acc, f1 = _calc_metrics(self.pred_labels, self.true_labels, self.dataset_configs.class_names)
        return acc, f1


    def train(self):
        copy_Files(self.exp_log_dir)  # save a copy of training files

        self.metrics = {'accuracy': [], 'f1_score': []}

        # fixing random seed
        fix_randomness(int(self.seed_id))

        # Logging
        self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.exp_log_dir, self.seed_id)
        self.logger.debug(self.hparams)

        # Load data
        self.load_data(self.dataset)

        model = FA_CTGTrans(configs=self.dataset_configs, hparams=self.hparams)
        model.to(self.device)
        # print(model)

        # Average meters
        loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

        # class_weight = get_class_weight(self.cw_dict)
        # print('class_weight:',class_weight)
        #
        # class_weights = torch.tensor(list(class_weight.values()), dtype=torch.float32).to(self.device)
        # print('class_weights:', class_weights)

        self.cross_entropy = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
            betas=(0.9, 0.99)
        )

        best_acc = 0
        best_f1 = 0
        best_loss=1.0
        best_qi=0

        # training..
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0

            self.pred_labels = []
            self.true_labels = []

            for step, batches in enumerate(self.train_dl):
                batches = to_device(batches, self.device)

                data = batches['samples'].float()
                labels = batches['labels'].long()

                self.optimizer.zero_grad()

                logits = model(data)

                loss_function = NRL_loss(alpha=0.5, beta=1, num_classes=2)
                x_ent_loss = loss_function(logits,labels)

                train_loss += x_ent_loss.item()

                _, predicted = torch.max(logits.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                x_ent_loss.backward()
                self.optimizer.step()

                losses = {'Total_Tra_loss': x_ent_loss.item()}

                for key, val in losses.items():
                    loss_avg_meters[key].update(val, self.hparams["batch_size"])

                self.pred_labels.extend(torch.argmax(logits, dim=1).tolist())
                self.true_labels.extend(labels.tolist())

            self.train_losses.append(train_loss / len(self.train_dl))
            self.train_accuracies.append(100 * correct_train / total_train)
            save_path = os.path.join(self.exp_log_dir, 'train_weights.pth')
            torch.save(model.state_dict(), save_path)
            tr_acc, tr_f1 = self.calc_results_per_run()
            self.logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in loss_avg_meters.items():
                self.logger.debug(f'{key}\t: {val.avg:2.4f}')
            self.logger.debug(f'TRAIN: Acc:{tr_acc:2.4f} \t F1:{tr_f1:2.4f}')

            # VALIDATION part
            val_loss,quality_index= self.evaluate(model, self.val_dl)
            self.val_losses.append(val_loss)

            correct_val, total_val = self.evaluate_accuracy(model, self.val_dl)
            self.val_accuracies.append(100 * correct_val / total_val)
            ts_acc, ts_f1 = self.calc_results_per_run()

            if quality_index > best_qi:
                best_qi = quality_index
                save_path = os.path.join(self.exp_log_dir, 'validation_best')
                torch.save(model.state_dict(), save_path)

                _save_metrics_val(self.pred_labels, self.true_labels, self.exp_log_dir, self.home_path,
                              self.dataset_configs.class_names)

            # logging
            self.logger.debug(f'VAL  : best_loss:{best_loss:2.4f}')
            self.logger.debug(f'VAL  : Acc:{ts_acc:2.4f} \t F1:{ts_f1:2.4f} (best: {best_f1:2.4f})')
            self.logger.debug(f'VAL  : 【best_qi:{best_qi:2.4f}】')
            self.logger.debug(f'-------------------------------------')

        # LAST EPOCH
        self.logger.debug("LAST EPOCH PERFORMANCE ...")
        self.logger.debug(f'Acc:{ts_acc:2.4f} \t F1:{ts_f1:2.4f}')

        self.logger.debug(":::::::::::::")
        # BEST EPOCH
        self.logger.debug("BEST EPOCH PERFORMANCE ...")
        self.logger.debug(f'Acc:{best_acc:2.4f} \t F1:{best_f1:2.4f}')

        self.plot_losses()
        plt.clf()
        self.plot_accuracy_curve()


    def evaluate_accuracy(self, model, dataset):
        model.to(self.device).eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batches in dataset:
                batches = to_device(batches, self.device)
                data = batches['samples'].float()
                labels = batches['labels'].long()

                # forward pass
                predictions = model(data)

                _, predicted = torch.max(predictions.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct, total

    def calculate_metrics(self,true_labels, pred_labels):
        cm = confusion_matrix(true_labels, pred_labels)

        tp = cm[1, 1]
        fp = cm[0, 1]
        tn = cm[0, 0]
        fn = cm[1, 0]

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        quality_index = (sensitivity * specificity) ** 0.5

        auc = roc_auc_score(true_labels, pred_labels)
        return sensitivity, specificity, quality_index, auc


    def plot_confusion_matrix(self,true_labels, pred_labels, class_names, save_dir):
        cm = confusion_matrix(true_labels, pred_labels)

        plt.figure(figsize=(10, 8))
        sns.set(font_scale=1.2)
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix_val.png'))  # 保存混淆矩阵图像
        plt.show()

    def evaluate(self, model, dataset):
        model.to(self.device).eval()

        total_loss_ = []
        total_loss = 0.0

        self.pred_labels = np.array([])
        self.true_labels = np.array([])

        with torch.no_grad():
            for batches in dataset:
                batches = to_device(batches, self.device)
                data = batches['samples'].float()
                labels = batches['labels'].long()

                # forward pass
                predictions = model(data)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss += loss.item()

                total_loss_.append(loss.item())
                pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability


                self.pred_labels = np.append(self.pred_labels, pred.cpu().numpy())
                self.true_labels = np.append(self.true_labels, labels.data.cpu().numpy())

        sensitivity, specificity, quality_index, auc = self.calculate_metrics(self.true_labels, self.pred_labels)
        self.logger.debug(f'Val_Sen: {sensitivity:.4f}')
        self.logger.debug(f'Val_Spe: {specificity:.4f}')
        self.logger.debug(f'Val_QI: {quality_index:.4f}')
        self.logger.debug(f'Val_AUC: {auc:.4f}')

        avg_loss = np.mean(total_loss_)
        self.logger.debug(f'Total_Val_loss: {avg_loss:.4f}')

        self.trg_loss = torch.tensor(total_loss_).mean()  # average loss

        avg_loss1 = total_loss / len(dataset)
        return avg_loss1,quality_index

    def plot_losses(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, label='Training Loss', color='green')
        plt.plot(epochs, self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curve')
        plt.legend()
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(self.exp_log_dir, f'losses_curve_{current_time}.png')
        plt.savefig(save_path)
        plt.show()

    def plot_accuracy_curve(self):
        epochs = range(1, len(self.train_accuracies) + 1)
        plt.plot(epochs, self.train_accuracies, label='Training Accuracy', color='green')
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy Curve')
        plt.legend()
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(self.exp_log_dir, f'accuracy_curve_{current_time}.png')
        plt.savefig(save_path)
        plt.show()



from torch.nn.modules.loss import _WeightedLoss


class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                              self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss

class NormalizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedCrossEntropy, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return self.scale * nce.mean()


class NRL_loss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=2):
        super(NRL_loss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = LabelSmoothCrossEntropyLoss(smoothing=0.1)
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)


    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min = 1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()

        return loss




