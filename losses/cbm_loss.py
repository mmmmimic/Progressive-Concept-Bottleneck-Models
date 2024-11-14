import torch.nn as nn
import torch
import numpy as np
import math

# class HCBMLoss:
#     def __init__(self, **kwargs):
#         # self.bce = nn.BCELoss()
#         self.bce = nn.BCEWithLogitsLoss(reduction='none')
#         # self.mse = nn.MSELoss(reduction='none')
#         self.mse = nn.SmoothL1Loss(reduction='none') # less sensitive to outliers than mse
#         self.mbce = nn.BCEWithLogitsLoss()
#         # self.mmse = nn.MSELoss()
#         self.mmse = nn.SmoothL1Loss()
#         self.cse = nn.CrossEntropyLoss()
#         self.act = lambda x: x#nn.Sigmoid()
    
#     def __call__(self, outs):
#         logits = outs['logit']
#         label = outs['label']
#         epoch = outs['epoch']

#         concept_logits = self.act(outs['concept_logit'])
#         concept_gt = outs['concept']
#         mask = outs['concept_mask']

#         batch_size = concept_logits.size(0)

#         # if epoch < 30:
#         #     loss = self.bce(concept_logits, concept_gt)*10
#         # else:
#         #     loss = self.cse(logits, label) + self.bce(concept_logits, concept_gt)
#         concept_logits[:,[0, 1, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 25, 26]] = torch.relu(concept_logits[:,[0, 1, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 25, 26]])
#         # concept_logits[:,[0, 1, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 25, 26]] = torch.sigmoid(concept_logits[:,[0, 1, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 25, 26]])

#         #loss = self.cse(logits, label)*0.01# + self.mbce(concept_logits[:,[2, 3, 4, 8, 13, 17, 22, 24]], concept_gt[:,[2, 3, 4, 8, 13, 17, 22, 24]]) + self.mmse(concept_logits[:,[0, 1, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 25, 26]], concept_gt[:,[0, 1, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 25, 26]]*10.)/10.
#         loss = self.cse(logits, label)*0.01# + self.mbce(concept_logits[:,[2, 3, 4, 8, 13, 17, 22, 24]], concept_gt[:,[2, 3, 4, 8, 13, 17, 22, 24]])

#         concept_loss = 0
#         cnter = 0
#         for i in [2, 3, 4, 8, 13, 17, 22, 24]:
#             ind = mask[:, i]
#             if not torch.sum(ind):
#                 continue
#             else:
#                 ind = ind > 0

#             concept = concept_gt[ind,i].clone() # [64]
#             concept_pred = concept_logits[ind,i].clone() # [64]

#             batch_loss = self.bce(concept_pred, concept)
#             count_1 = torch.sum(concept)
#             count_0 = batch_size - count_1
#             weight_0 = count_1 / batch_size
#             weight_1 = 1 - weight_0
#             # if weight_0 <= 1e-6:
#             #     weight_0 = 1e-6
#             # if weight_1 <= 1e-6:
#             #     weight_1 = 1e-6
#             weight = (1-concept)*weight_0 + concept*weight_1
#             weight = weight / torch.sum(weight)
#             batch_loss = batch_loss * weight
#             batch_loss = torch.sum(batch_loss)

#             # batch_loss = nn.BCEWithLogitsLoss()(concept_pred, concept)
#             concept_loss = concept_loss + batch_loss
#             cnter += 1
        
#         concept_loss = concept_loss / cnter
#         loss = loss + concept_loss*10


#         concept_loss = 0
#         cnter = 0
#         for i in [0, 1, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 25, 26]:
#             ind = mask[:, i]
#             if not torch.sum(ind):
#                 continue
#             else:
#                 ind = ind > 0
            
#             concept = concept_gt[ind,i].clone()
#             concept_pred = concept_logits[ind,i].clone()
#             # oversampling according to the density
#             # count density
#             density = torch.zeros_like(concept) # [N]
#             for c in range(density.size(0)):
#                 density[c] = torch.sum(concept == concept[c])
#             weight = torch.sum(density) - density
#             weight = weight / torch.sum(weight)
#             # weight = torch.softmax(-density, dim=0)
#             # print(concept, density, weight)

#             batch_loss = self.mse(concept_pred, concept*10.) # [64]
#             # density = torch.zeros_like(concept)

#             # for j in range(concept.size(0)):
#             #     v = concept[j]
#             #     h = 0.01
#             #     # gaussian kernel
#             #     ds = torch.sum(torch.exp(-((v - concept)/h)**2/2)) / (batch_size*h*math.sqrt(2*math.pi))
#             #     density[j] = ds
#             # min_ds = torch.min(density).item()
#             # max_ds = torch.max(density).item()

#             # if (max_ds - min_ds):
#             #     density = (density - min_ds) / (max_ds - min_ds)
#             # else:
#             #     density = torch.ones_like(density)
#             # alpha = 1
#             # eps = 1e-6
#             # weight = 1 - alpha*density
#             # weight = torch.relu(weight - eps) + eps
#             # weight /= torch.sum(weight)

#             batch_loss = batch_loss * weight
#             batch_loss = torch.sum(batch_loss)

#             # concept_pred = torch.clamp(concept_pred, max=10)
#             # batch_loss = nn.BCELoss(weight=weight)(concept_pred/10., concept)
#             concept_loss = concept_loss + batch_loss#/10.
#             cnter += 1
#         concept_loss /= cnter
#         # for i, j in zip(concept, weight):
#         #     print(i, j)

#         loss = loss + concept_loss

#         return loss

class PCBMLoss:
    def __init__(self, **kwargs):
        self.cse = nn.CrossEntropyLoss(label_smoothing=0.05)
        # self.cse = nn.BCEWithLogitsLoss()
        self.bce = nn.BCELoss()

    def __call__(self, outs):
        logits = outs['logit']
        label = outs['label']
        emb = outs['emb']
        plane_pred = outs['plane_pred_seg']
        # label_onehot = torch.nn.functional.one_hot(label, num_classes=logits.size(1)).float()
        # # label smoothing
        # label_onehot = label_onehot - 0.18
        # label_onehot = torch.relu(label_onehot) + 0.02

        # crossentropy loss for classification
        # # avoid over-confident
        # norms = torch.norm(logits, p=2, dim=-1, keepdim=True) + 1e-7
        # logits = logits / norms

        cse_loss = self.cse(logits, label) # focal loss also helps alleviate over-confidence
        confidency_penalty = torch.sum(-torch.nn.functional.log_softmax(logits, dim=1), dim=1).mean()*0.01
        # confidency_penalty = 0

        # # ranking loss for ranking the confidence (from RankNet)
        # # we expect the confidence ranking to be:
        # # if SP: SP > NSP > Others
        # # if NSP: NSP > SP > Others
        # # for the same class: SP > NSP > Others
        # prob = torch.softmax(logits, dim=1)
        # # prob = torch.sigmoid(logits)
        # # prob = torch.zeros_like(logits)
        # class_num = 3

        # bce
        # label_onehot = torch.nn.functional.one_hot(label, num_classes=logits.size(1)).float()
        # prob = torch.sigmoid(logits)
        # prob[:,-1] = torch.softmax(logits, dim=1)[:,-1]
        # for i in range(class_num):
            # p = torch.softmax(logits[:,[i,i+class_num,-1]], dim=1)
            # prob[:,[i,i+class_num]] = p[:,[0,1]]
            # confidency_penalty += torch.sum(-torch.nn.functional.log_softmax(logits[:,[i,i+class_num,-1]], dim=1), dim=1).mean()*0.1
        # cse_loss = nn.BCELoss()(prob, label_onehot)

        # ranking_loss = 0
        # # emb (B, 512), generate pesudo image quality label

        # for i in range(class_num):
        #     # positive_emb = emb[label==i, :]
        #     # negative_emb = emb[label==i+class_num, :]
        #     # positive_dist = torch.cdist(emb.unsqueeze(0), positive_emb.unsqueeze(0), p=2.0).squeeze(0)
        #     # negative_dist = torch.cdist(emb.unsqueeze(0), negative_emb.unsqueeze(0), p=2.0).squeeze(0)

        #     # # for positive samples
        #     # # ranking_label = torch.zeros_like(label)
        #     # # ranking_label = torch.mean(negative_dist, dim=1)
        #     # # ranking_label = ranking_label[label==i]
        #     # ranking_label = torch.mean(negative_dist, dim=1) - torch.mean(positive_dist, dim=1)

        #     # sign = torch.sign(ranking_label.unsqueeze(0) - ranking_label.unsqueeze(1))
        #     # diff = prob[:, i].unsqueeze(0) - prob[:, i].unsqueeze(1)
        #     # # diff = prob[label==i, i].unsqueeze(0) - prob[label==i, i].unsqueeze(1)
        #     # diff = sign * diff
        #     # index = ranking_label!=0
        #     # if torch.sum(index):
        #     #     ranking_loss += -torch.relu(diff[index] + 0.1).mean()/2#-torch.nn.LogSigmoid()(diff[index]*100).mean()/2
                
        #     # # # for negative samples
        #     # # ranking_label = torch.zeros_like(label)
        #     # ranking_label = -torch.mean(positive_dist, dim=1)
        #     # ranking_label = ranking_label[label==i+4]
        #     # # ranking_label = torch.mean(negative_dist, dim=1) - torch.mean(positive_dist, dim=1)

        #     # sign = torch.sign(ranking_label.unsqueeze(0) - ranking_label.unsqueeze(1))
        #     # # diff = prob[:, i].unsqueeze(0) - prob[:, i].unsqueeze(1)
        #     # diff = prob[label==i+4, i].unsqueeze(0) - prob[label==i+4, i].unsqueeze(1)
        #     # diff = sign * diff
        #     # index = ranking_label!=0
        #     # if torch.sum(index):
        #     #     ranking_loss += -torch.nn.LogSigmoid()(diff[index]*100).mean()    
            
        #     ranking_label = torch.zeros_like(label)
        #     ranking_label[label==i] = 2
        #     ranking_label[label==i+class_num] = 1   
        #     # sign = torch.sign(ranking_label.unsqueeze(0) - ranking_label.unsqueeze(1))
        #     # diff = prob[:, i].unsqueeze(0) - prob[:, i].unsqueeze(1)
        #     # diff = sign * diff
        #     # index = ranking_label!=0
        #     # if torch.sum(index):
        #     #     ranking_loss += -torch.nn.LogSigmoid()(diff[index]*100).mean()

        #     sign = -torch.sign(ranking_label.unsqueeze(0) - ranking_label.unsqueeze(1))
        #     diff = prob[:, i].unsqueeze(0) - prob[:, i].unsqueeze(1)
        #     diff = sign * diff
        #     index = ranking_label!=0
        #     if torch.sum(index):
        #         ranking_loss += -torch.relu(diff[index] + 0.3).sum()/2/sign.size(0)

        #     # entropy_pos = -torch.sum(torch.nn.functional.log_softmax(prob[label==i, i], dim=0)) if torch.sum(label==i) else ranking_loss*0.
        #     # entropy_neg = -torch.sum(torch.nn.functional.log_softmax(prob[label==i+class_num, i], dim=0)) if torch.sum(label==i+class_num) else ranking_loss*0.
        #     # ranking_loss +=  100 / (entropy_pos + entropy_neg + 1) if (entropy_pos + entropy_neg) else 0

        # ranking_loss /= 10

        # print(ranking_loss)
        return cse_loss + confidency_penalty# + ranking_loss# + confidency_penalty

class HCBMLossCUB:
    def __init__(self, **kwargs):
        # self.bce = nn.BCELoss()
        self.bce = []
        self.cse = nn.CrossEntropyLoss()
        self.weight = [10.783783783783784, 15.76923076923077, 1.463276836158192, 3.116738197424893, 3.0851788756388414, 3.4613953488372093, 14.724590163934426, 1.4136889783593358, 3.8888888888888893, 6.773095623987034, 3.3560399636693914, 2.815433571996818, 11.050251256281408, 1.430816016218956, 4.68920521945433, 7.08768971332209, 16.06761565836299, 8.787755102040816, 6.001459854014598, 5.542974079126876, 1.339512195121951, 5.97093023255814, 0.5426182052106787, 12.742120343839542, 11.587926509186351, 3.4325323475046208, 2.993338884263114, 18.03174603174603, 2.6610687022900765, 7.2832469775474955, 8.223076923076922, 2.4981765134938003, 4.9503722084367245, 4.648998822143699, 2.2361673414304994, 7.25473321858864, 11.020050125313283, 19.06694560669456, 3.5938697318007664, 14.571428571428571, 9.949771689497718, 5.754929577464789, 5.10178117048346, 1.7034949267192783, 7.384615384615385, 19.236286919831223, 8.3671875, 4.011494252873563, 1.6943820224719102, 14.470967741935484, 0.04306220095693769, 1.945945945945946, 0.5812726673260797, 19.06694560669456, 6.288753799392097, 7.8161764705882355, 12.032608695652174, 1.9441375076734193, 10.726161369193154, 5.642659279778393, 5.764456981664316, 1.570203644158628, 5.188387096774194, 10.841975308641976, 6.862295081967213, 5.125159642401021, 15.481099656357387, 3.2745098039215685, 4.328888888888889, 10.841975308641976, 17.881889763779526, 10.640776699029127, 5.754929577464789, 7.08768971332209, 1.43823080833757, 7.720000000000001, 1.3683950617283949, 5.764456981664316, 0.5960066555740433, 4.346711259754738, 9.72930648769575, 18.736625514403293, 0.6833976833976834, 1.0176693310896088, 7.751824817518248, 13.231454005934719, 1.3057692307692306, 14.62214983713355, 6.862295081967213, 0.33593314763231197, 4.3706606942889135, 3.729783037475345, 6.823817292006526, 2.526470588235294, 3.8839103869653764, 10.231850117096018, 3.412143514259429, 2.6278366111951588, 8.03201506591337, 7.156462585034014, 1.0279069767441862, 17.66147859922179, 19.06694560669456, 6.052941176470588, 7.832412523020258, 17.09811320754717, 1.9550215650030807, 12.285318559556787, 2.709203402938902, 18.900414937759336, 4.032528856243442, 3.5632730732635585]
        for i in range(112):
            self.bce.append(nn.BCEWithLogitsLoss(weight=torch.tensor(self.weight[i]).cuda()))

    def __call__(self, outs):
        logits = outs['logit']
        label = outs['label']

        concept_logits = outs['concept_logit']
        concept_gt = outs['concept']
        if 'aux_logit' in outs.keys():
            aux_logits = outs['aux_logit']
        else:
            aux_logits = None

        # loss = self.cse(logits, label) + self.bce(concept_logits, concept_gt)*100
        # loss = self.cse(logits, label)
        loss = 0
        for i in range(112):
            loss += self.bce[i](concept_logits[:,i], concept_gt[:,i])
            loss += self.bce[i](aux_logits[:,i], concept_gt[:,i])*0.4 if aux_logits is not None else 0
        loss = loss / 112

        loss = loss + self.cse(logits, label)*0.1
        return loss