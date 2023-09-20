import configargparse
import data_loader
import os
import torch
import models1
import utils
from utils import str2bool
import numpy as np
import random
from sklearn.metrics import recall_score
from sklearn.preprocessing import MultiLabelBinarizer

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # network related
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--src_domain', type=str, required=True)
    parser.add_argument('--tgt_domain', type=str, required=True)
    parser.add_argument('--src_val', type=str, required=True)
    # training related
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False, help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=10)
    parser.add_argument('--transfer_loss', type=str, default='mmd')
    return parser

def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    folder_val = os.path.join(args.data_dir, args.src_val)
    source_loader, n_class = data_loader.load_data(
        folder_src, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
    target_train_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
    source_test_loader, _ = data_loader.load_data(
        folder_val, args.batch_size, infinite_data_loader=False , train=False, num_workers=args.num_workers)
    return source_loader, target_train_loader, source_test_loader, n_class

def get_model(args):
    model = models1.TransferNet(
        args.n_class, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter, use_bottleneck=args.use_bottleneck).to(args.device)
    return model

def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    return optimizer

def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler


def compute_uar(y_true, y_pred):
    unique_labels = np.unique(y_true)  # 获取唯一的标签类别
    recalls = []
    for label in unique_labels:
        true_positives = np.sum((y_true == label) & (y_pred == label))
        possible_positives = np.sum(y_true == label)
        recall = true_positives / possible_positives if possible_positives > 0 else 0
        recalls.append(recall)
    
    uar = np.mean(recalls)
    return uar

def test(model, source_test_loader, args):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_source_dataset = len(source_test_loader.dataset)
    pred_label = []
    true_label = []
    with torch.no_grad():
        for data, source in source_test_loader:
            data, source = data.to(args.device), source.to(args.device)
            s_output = model.predict(data)
            loss = criterion(s_output, source)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            pred_label.extend(pred.tolist())
            true_label.extend(source.tolist())
            correct += torch.sum(pred == source)
    acc = 100. * correct / len_source_dataset
    uar = compute_uar(true_label, pred_label)
    print(len_source_dataset)
    return acc,  uar,  test_loss.avg, pred_label, true_label

def train(source_loader, target_train_loader, source_test_loader, model, optimizer, lr_scheduler, args):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch 
    
    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    best_acc = 0
    best_uar = 0
    stop = 0
    log = []
    final_pred_label = []
    final_true_label = []
    for e in range(1, args.n_epoch+1):
        model.train()
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.epoch_based_processing(n_batch)
        
        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)
        
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(n_batch):
            data_source, label_source = next(iter_source) # .next()
            data_target, label_target = next(iter_target) # .next()
            #data_target, _ = next(iter_target) # .next()
            data_source, label_source = data_source.to(args.device), label_source.to(args.device)
            #data_target = data_target.to(args.device)
            data_target , label_target= data_target.to(args.device), label_target.to(args.device)
            clf_loss, transfer_loss = model(data_source, data_target, label_source,label_target)
            loss = clf_loss + args.transfer_loss_weight * transfer_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
            
        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg])
        
        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}'.format(
                        e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg)
        # Test
        stop += 1
        test_acc, uar, test_loss, pred_label, true_label = test(model, source_test_loader, args)
        info += ', test_loss {:4f}, test_acc: {:.4f}, test_uar: {:.4f}'.format(test_loss, test_acc,  uar)
        np_log = np.array(log, dtype=float)
        np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
        if best_acc < test_acc:
            best_acc = test_acc
            best_uar = uar
            final_pred_label = pred_label
            final_true_label = true_label
            stop = 0
        if args.early_stop > 0 and stop >= args.early_stop:
            print(info)
            break
        print(info)
    final_true_label = [str(i)+'\n' for i in final_true_label]
    f=open("true_label.txt","w")
    f.writelines(final_true_label)
    f.close()
    final_pred_label = [str(i)+'\n' for i in final_pred_label]
    f1=open("pred_label.txt","w")
    f1.writelines(final_pred_label)
    f1.close()
    print('Transfer result: {:.4f},  {:.4f}'.format(best_acc, best_uar))

def main():
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(args)
    set_random_seed(args.seed)
    source_loader, target_train_loader, source_test_loader, n_class = load_data(args)
    setattr(args, "n_class", n_class)
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    model = get_model(args)
    optimizer = get_optimizer(model, args)
    
    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
    train(source_loader, target_train_loader, source_test_loader, model, optimizer, scheduler, args)
    

if __name__ == "__main__":
    main()
