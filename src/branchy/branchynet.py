import time

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from scipy.stats import entropy

from util import accuracy, do_exit_statistics, write_into_csv


class BranchyNet(nn.Module):
    def __init__(self, network, device, weight_list=None, thresholdExits=None, percentTestExits=.9, percentTrainKeeps=.5,
                 lr_main=0.01,lr_branches=0.01, momentum=0.9, weight_decay=8e-4, alpha=0.01,
                 confidence_metric_txt="confidence",opt="Adam", joint=True, verbose=False):
        super().__init__()
        self.network = network
        self.lr_main = lr_main
        self.lr_branches = lr_branches
        self.momentum = momentum
        self.opt = opt
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.joint = joint
        self.verbose = verbose
        self.thresholdExits = thresholdExits
        self.percentTestExits = percentTestExits
        self.percentTrainKeeps = percentTrainKeeps
        self.gpu = False
        self.criterion = nn.CrossEntropyLoss()
        self.weight_list = weight_list
        self.device = device
        self.setMetric(confidence_metric_txt)
        steps = 1
        self.setOptimizer(steps)
        if (weight_list is None):
            # self.weight_list = np.ones(len(self.network.stages))
            self.weight_list = np.linspace(1, 0.1, len(self.network.stages))



    def setOptimizer(self, steps):

        if self.opt == "Adam":
            self.optimizer_main = optim.Adam([{"params": self.network.stages.parameters()},
                                              {"params": self.network.classifier.parameters()}], lr=self.lr_main,
                                             betas=(0.9, 0.999), eps=1e-08,
                                             weight_decay=self.weight_decay)

        else:
            self.optimizer_main = optim.SGD([{"params": self.network.stages.parameters()},
                                             {"params": self.network.classifier.parameters()}], lr=self.lr_main,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay)

        self.scheduler_main = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_main, steps, eta_min=0, last_epoch=-1,
                                                                   verbose=True)

        self.optimizer_list = []
        self.scheduler_list = []

        for i in range(len(self.network.stages)):
            if i == len(self.network.stages) - 1:
                opt_branch = optim.SGD([{"params": self.network.stages.parameters()},
                                        {"params": self.network.classifier.parameters()}], lr=self.lr_branches,
                                       momentum=self.momentum,
                                       weight_decay=self.weight_decay)

            else:
                opt_branch = optim.SGD([{"params": self.network.stages[i].parameters()},
                                        {"params": self.network.exits.parameters()}], lr=self.lr_branches,
                                       momentum=self.momentum,
                                       weight_decay=self.weight_decay)

            self.optimizer_list.append(opt_branch)
            scheduler_branches = optim.lr_scheduler.CosineAnnealingLR(opt_branch, steps, eta_min=0, last_epoch=-1,
                                                                      verbose=True)
            self.scheduler_list.append(scheduler_branches)

    def setMetric(self, metric):
        if metric == 'confidence':
            self.confidence_metric = self.compute_confidence
            self.shouldExist = self.verify_confidence
        elif metric == 'entropy':
            self.confidence_metric = self.compute_entropy()
            self.shouldExist = self.verify_entropy()
        elif self.confidence_metric_txt == "softmargin":
            self.confidence_metric = self.compute_sm
            self.shouldExist = self.verify_sm
        else:
            raise NotImplementedError("This confidence metric is not supported.")

    def training_123(self):
        self.network.stages.train()
        self.network.exits.train()
        self.network.classifier.train()

    def named_parameters(self):
        return [self.network.stages.named_parameters(), self.network.exits.named_parameters()]

    def testing(self):

        self.network.stages.eval()
        self.network.exits.eval()
        self.network.classifier.eval()

    def to_gpu(self):
        self.gpu = True
        self.network = self.network.cuda()

    def to_cpu(self):
        self.gpu = False
        self.network = self.network.to("cpu")

    @property
    def threshold(self):
        return self.thresholdExits

    @threshold.setter
    def threshold(self, t):
        self.thresholdExits = t

    def compute_sm(self, softmax_output):
        maxs, _ = torch.topk(softmax_output, 2)
        return maxs[0] - maxs[1]

    def compute_entropy(self, softmax_output):
        entropy_value = np.array([entropy(output) for output in softmax_output.cpu().detach().numpy()])
        return entropy_value

    def verify_entropy(self, entropy_value, thresholdExitsValue):
        return entropy_value <= thresholdExitsValue

    def verify_confidence(self, confidence_value, thresholdExitsValue):
        return confidence_value >= thresholdExitsValue

    def verify_sm(self, confidence_value, thresholdExitsValue):
        return confidence_value >= thresholdExitsValue

    def compute_confidence(self, softmax_output):
        # print("compute_confidence:")
        confidence_value, _ = torch.max(softmax_output, 1)
        # print("confidence_value:", confidence_value)
        return confidence_value.cpu().detach().numpy()

    def train_main(self, x, t, adj, train_mask):
        self.training_123()
        self.optimizer_main.zero_grad()
        output, infered_class = self.network.forwardMain(x, adj)
        loss = self.criterion(output[train_mask], t[train_mask])
        loss.backward()
        self.optimizer_main.step()

        acc1 = accuracy(output[train_mask], t[train_mask])
        return loss.item(), acc1

    def val_main(self, x, t, adj, valid_mask):
        self.testing()
        with torch.no_grad():
            output, infered_class, sss = self.network.forwardMain1(x, adj)
            loss = self.criterion(output[valid_mask], t[valid_mask])
            acc1 = accuracy(output[valid_mask], t[valid_mask])
        return loss.item(), acc1, output[valid_mask]

    def test_main(self, x, t, adj, test_mask):
        output, infered_class = self.network.forwardMain(x, adj)
        loss = self.criterion(output[test_mask], t[test_mask])
        acc1 = accuracy(output[test_mask], t[test_mask])
        return loss.item(), acc1

    def train_branches(self, x, t, adj, train_mask):
        remainingXVar = x
        remainingTVar = t
        numexits, losses, acc_list = [], [], []
        n_models = len(self.network.stages)
        n_samples = x.data.shape[0]
        softmax = nn.Softmax(dim=1)
        [model.zero_grad() for model in self.network.stages]
        self.training_123()
        for i in range(n_models):
            if (remainingXVar is None) or (remainingTVar is None):
                break
            self.optimizer_list[i].zero_grad()
            output_branch, class_infered_branch = self.network.forwardBranchesTrain(remainingXVar, i, adj)
            output_branch = output_branch[train_mask]
            class_infered_branch = class_infered_branch[train_mask]
            loss_branch = self.criterion(output_branch, remainingTVar)
            acc_branch=accuracy(output_branch,remainingTVar)


            losses.append(loss_branch)
            acc_list.append(acc_branch)
            softmax_output = softmax(output_branch)
            confidence_measure = self.confidence_metric(softmax_output)
            total = confidence_measure.shape[0]
            idx = np.zeros(total, dtype=bool)
            if i == n_models - 1:
                idx = np.ones(confidence_measure.shape[0], dtype=bool)
                numexit = sum(idx)
            else:
                if self.thresholdExits is not None:
                    min_ent = 0
                    if isinstance(self.thresholdExits, list):
                        idx[self.shouldExist(confidence_measure, self.thresholdExits[i])] = True
                        numexit = sum(idx)
                    else:
                        idx[self.shouldExist(confidence_measure, self.thresholdExits)] = True
                        numexit = sum(idx)

                else:  # please see
                    if (isinstance(self.percentTrainKeeps, list)):
                        numkeep = int((self.percentTrainKeeps[i]) * n_samples)
                        numexit = int(total - numkeep)
                    else:
                        numkeep = int(self.percentTrainKeeps * confidence_measure.shape[0])
                    numexit = int(total - numkeep)
                    esorted = confidence_measure.argsort()
                    idx[esorted[:numexit]] = True

            # here
            total = confidence_measure.shape[0]
            numkeep = total - numexit
            numexits.append(numexit)
            xdata = remainingXVar.data
            tdata = remainingTVar.data
            if numkeep > 0:
                xdata_keep = xdata
                tdata_keep = tdata
                remainingXVar = Variable(xdata_keep, requires_grad=False).to(self.device)
                remainingTVar = Variable(tdata_keep, requires_grad=False).to(self.device)
        for i, (weight, loss) in enumerate(zip(self.weight_list, losses)):
            loss = weight * loss
            loss.backward()
        self.optimizer_main.step()
        [optimizer.step() for optimizer in self.optimizer_list]
        losses = np.array([loss.item() for loss in losses])
        acc_list = np.array(acc_list)
        return losses, acc_list, numexits

    def setRemainingMask(self, idx, mask):
        r_mask = mask.detach().clone()
        k = 0
        for i, x in enumerate(r_mask):
            if x:
                if idx[k]:
                    r_mask[i] = False
                k += 1
        return r_mask

    def setCopyMask(self, idx, mask):
        c, count = 0, mask.sum()
        for i, x in enumerate(idx):
            if count > c and mask[i]:
                idx[i] = True
                c += 1
            else:
                idx[i] = False
        return idx

    def howManyToExit(self, confidence_measure, n_models, present_model, n_samples):
        idx = np.zeros(confidence_measure.shape[0], dtype=bool)
        i = present_model
        if present_model == n_models - 1:
            print("# Always exit all at end!")
            idx = np.ones(confidence_measure.shape[0], dtype=bool)  # Always exit all at end!
            numexit = sum(idx)
        else:
            if self.thresholdExits is not None:
                min_ent = 0
                if isinstance(self.thresholdExits, list):
                    idx[self.shouldExist(confidence_measure, self.thresholdExits[i])] = True
                    numexit = sum(idx)
                else:
                    idx[self.shouldExist(confidence_measure, self.thresholdExits)] = True
                    numexit = sum(idx)

            else:
                if isinstance(self.percentTestExits, list):
                    numexit = int((self.percentTestExits[i]) * n_samples)
                else:
                    numexit = int(self.percentTestExits * confidence_measure.shape[0])

                esorted = confidence_measure.argsort()
                idx[esorted[:numexit]] = True
        return idx, numexit

    @torch.no_grad()
    def val_branches(self, x, t, adj, mask):
        self.testing()
        remainingMask = mask
        numexits, losses, acc_list, acc_branches_list,output_list = [], [], [], [],[]
        branches_time={}
        n_models = len(self.network.stages)
        n_samples = x.data.shape[0]
        softmax = nn.Softmax(dim=1)
        sigmoid = nn.Sigmoid()
        idx_ = np.zeros(mask.sum(), dtype=bool)
        time_ = time.time()
        branches_time[100]=time_
        for i in range(n_models):
            if remainingMask is None:
                acc_branches_list.append(0.)
                losses.append(0.)
                numexits.append(0)
                output_list.append(0)
                continue
            output_branch1, class_infered_branch1 = self.network.forwardBranchesTrain(x, i, adj, False)
            output_branch, class_infered_branch = output_branch1[remainingMask], class_infered_branch1[remainingMask]
            softmax_output = softmax(output_branch)
            confidence_measure = self.confidence_metric(softmax_output)


            idx, numexit = self.howManyToExit(confidence_measure, n_models, i, n_samples)
            total = confidence_measure.shape[0]
            numkeep = total - numexit
            numexits.append(numexit)
            # print(numexits)
            if numkeep > 0:
                mask_to_exit = self.setRemainingMask(~idx, remainingMask)
                tVar_mask_to_exit = self.setCopyMask(idx_, idx)
                remainingMask = self.setRemainingMask(idx, remainingMask)
            else:
                mask_to_exit = self.setRemainingMask(~idx, remainingMask)
                tVar_mask_to_exit = self.setCopyMask(idx_, idx)
                remainingMask = None
            if numexit > 0:
                tt_ = time.time()
                exit_output, class_infered_branch = self.network.forwardBranchesTrain(x, i, adj)
                exit_output, class_infered_branch = exit_output[mask_to_exit], class_infered_branch[mask_to_exit]
                output_list.append(exit_output)
                exitTVar = t[tVar_mask_to_exit]
                print(class_infered_branch.shape,exitTVar.shape,exitTVar.size(0),class_infered_branch.size(),tVar_mask_to_exit.sum())
                # self.do_exit_statistics1(class_infered_branch, exitTVar)
                accuracy_branch = accuracy(exit_output, exitTVar)
                # accuracy_branch = 100 * class_infered_branch.eq(
                #     exitTVar.view_as(class_infered_branch)).sum().item() / exitTVar.size(0)

                acc_branches_list.append(accuracy_branch)
                loss = self.criterion(exit_output, exitTVar)
                losses.append(loss)
                tt_ = time.time() - tt_
                branches_time[i] = tt_
            else:
                acc_branches_list.append(0.), losses.append(0.)

        t_ = time.time() - time_
        branches_time[101] = t_
        write_into_csv("results/csv/outputs.csv",[branches_time])
        print("val_branches:time", t_)
        return do_exit_statistics(n_models, losses, acc_branches_list, numexits,t_)


    @torch.no_grad()
    def validate_branches_without_exit(self, x, t, adj, mask):
        self.testing()
        remainingMask = mask
        numexits, losses, acc_list, acc_branches_list = [], [], [], []
        n_models = len(self.network.stages)
        n_samples = x.data.shape[0]
        softmax = nn.Softmax(dim=1)
        idx_ = np.zeros(mask.sum(), dtype=bool)
        for i in range(n_models):
            # print("--------------------------")
            if remainingMask is None:
                acc_branches_list.append(0.)
                losses.append(0.)
                numexits.append(0)
                continue
            output_branch, class_infered_branch = self.network.forwardBranchesTrain(x, i, adj, False)
            output_branch, class_infered_branch = output_branch[remainingMask], class_infered_branch[remainingMask]

            # print("class_infered_branch:",class_infered_branch)
            # print("output_branch:",output_branch)
            confidence_measure = self.confidence_metric(softmax(output_branch))

            idx, numexit = self.howManyToExit(confidence_measure, n_models, i, n_samples)
            # print("idx, numexit :",idx.sum(), numexit )
            total = confidence_measure.shape[0]
            numkeep = total - numexit
            numexits.append(numexit)
            # print("total,numexit,numkeep:", total, numexit, numkeep)
            accuracy_branch = 100 * class_infered_branch.eq(
                t.view_as(class_infered_branch)).sum().item() / t.size(0)
            acc_branches_list.append(accuracy_branch)
            loss = self.criterion(output_branch, t)
            losses.append(loss)

            continue
            if numkeep > 0:
                mask_to_exit = self.setRemainingMask(~idx, remainingMask)
                tVar_mask_to_exit = self.setCopyMask(idx_, idx)
                remainingMask = self.setRemainingMask(idx, remainingMask)
            else:
                mask_to_exit = self.setRemainingMask(~idx, remainingMask)
                tVar_mask_to_exit = self.setCopyMask(idx_, idx)
                remainingMask = None
            if numexit > 0:
                exit_output, class_infered_branch = self.network.forwardBranchesTrain(x, i, adj)
                exit_output, class_infered_branch = exit_output[mask_to_exit], class_infered_branch[mask_to_exit]
                exitTVar = t[tVar_mask_to_exit]
                torch.set_printoptions(profile="full")
                # print("exit_output:",exit_output)
                # print("softmax:", softmax(exit_output))
                # print("exitTVar:",exitTVar)
                # print("class_infered_branch:",class_infered_branch)
                torch.set_printoptions(profile="default")
                accuracy_branch = 100 * class_infered_branch.eq(
                    exitTVar.view_as(class_infered_branch)).sum().item() / exitTVar.size(0)
                acc_branches_list.append(accuracy_branch)
                loss = self.criterion(exit_output, exitTVar)
                losses.append(loss)
            else:
                acc_branches_list.append(0.), losses.append(0.)
            # print("---------------------------------------------")

        return self.do_exit_statistics(n_models, losses, acc_branches_list, numexits)

    @torch.no_grad()
    def evaluate_branches(self, x, t, adj, mask):
        self.testing()
        predictions, true_labels = [], []
        softmax = nn.Softmax(dim=1)
        n_models = len(self.network.stages)
        for i in range(n_models):
            exit_output, class_infered_branch = self.network.forwardBranchesTrain(x, i, adj)
            predictions.append(softmax(exit_output[mask]))
            true_labels.append(t[mask])
        return predictions, true_labels

    def val_branchy_model(self, x, adj, mask):
        self.testing()
        remainingMask = mask
        n_models = len(self.network.stages)
        n_samples = x.data.shape[0]
        softmax = nn.Softmax(dim=1)
        time_ = time.time()
        for i in range(n_models):
            if remainingMask is None:
                break
            output_branch1, class_infered_branch1 = self.network.forwardBranchesTrain(x, i, adj, False)
            output_branch, class_infered_branch = output_branch1[remainingMask], class_infered_branch1[remainingMask]
            softmax_output = softmax(output_branch)
            confidence_measure = self.confidence_metric(softmax_output)
            idx, numexit = self.howManyToExit(confidence_measure, n_models, i, n_samples)
            total = confidence_measure.shape[0]
            numkeep = total - numexit
            if numkeep > 0:
                remainingMask = self.setRemainingMask(idx, remainingMask)
            else:
                remainingMask = None
        print("val_branchy_model-time", time.time() - time_)
        return
