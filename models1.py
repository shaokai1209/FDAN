import torch
import torch.nn as nn
from transfer_losses import TransferLoss
import backbones
from module import Attention, PreNorm, FeedForward, CrossAttention

class TransferNet(nn.Module):
    def __init__(self, num_class, base_net='resnet34', transfer_loss='mmd', use_bottleneck=True, bottleneck_width=256, max_iter=1000, **kwargs):
        super(TransferNet, self).__init__()
        self.num_class = num_class
        self.base_network = backbones.get_backbone(base_net)
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()
        
        self.classifier_layer = nn.Linear(feature_dim, num_class)
        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss()
        # cross-attention
        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(1):
            self.cross_attn_layers.append(nn.ModuleList([
                nn.Linear(256,256),
                nn.Linear(256, 256),
                PreNorm(256, CrossAttention(256, heads =2, dim_head = 32, dropout = 0)),
                nn.Linear(256, 256),
                nn.Linear(256, 256),
                PreNorm(256, CrossAttention(256, heads =2, dim_head = 32, dropout = 0)),
            ]))
    def forward(self, source, target, source_label, target_label):
        source = self.base_network(source)
        target = self.base_network(target)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)
        # cross-attention
        source = source[None,:]
        target = target[None,:]
        xs = source
        xl = target
        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:
            small_class = xs[:, 0]
            x_small = xs[:, 1:]
            large_class = xl[:, 0]
            x_large = xl[:, 1:]

            # Cross Attn for target

            cal_q = f_ls(large_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_small), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xl = torch.cat((cal_out, x_large), dim=1)

            # Cross Attn for source
            cal_q = f_sl(small_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_small), dim=1)
        source = xs.squeeze() # 32,256
        target = xl.squeeze()
        # classification
        source_clf = self.classifier_layer(source)
        target_clf = self.classifier_layer(target)
        clf_loss = self.criterion(source_clf, source_label) + self.criterion( target_clf, target_label) #  
        # transfer
        kwargs = {}
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            kwargs['target_label'] = target_label
            target_clf = self.classifier_layer(target)
            #kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
            aa = torch.nn.functional.softmax(target_clf, dim=1)
            #print("######################")
            #print("target_label:",target_label[:5])
            #print("label_pro",target_clf[:5])
            for i in range(len(aa)):
                for j in range(len(aa[i])):
                    if j==target_label[i]:
                        aa[i][j] = 1.0
                    else:
                        aa[i][j] = 0.0
            #print("label_matrix",aa[:5])
            kwargs['target_logits'] =aa
        
        transfer_loss = self.adapt_loss(source, target, **kwargs)
        return clf_loss, transfer_loss
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
            )
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        features = self.base_network(x)
        x = self.bottleneck_layer(features)
        clf = self.classifier_layer(x)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass
