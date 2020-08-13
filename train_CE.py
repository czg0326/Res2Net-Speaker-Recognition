#from __future__ import print_function
#torch
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.utils as utils

#sys
import numpy as np
import time,random,os,sys
import matplotlib.pyplot as plot
import argparse

#my
from utils.EER import EER,cosine,EER_kaldi
from utils.load_kaldi import TranLoader
from model_softmax import ResNetTimePooling, ResNetTimePooling_, Res2Net

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# data path
parser.add_argument('--dataroot', type=str, default='/nobackup/s3/projects/srlr/yaoshengyu/vox1/wav',#'/nobackup/s1/srlr/yaoshengyu/data/wav_vad',
                    help='path to dataset')
parser.add_argument('--log-dir', default='./data/pytorch_speaker_logs',
                    help='folder to output model checkpoints')
parser.add_argument('--resume',
                    default=None,
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=40, metavar='E',
                    help='number of epochs to train (default: 10)')
# Training options
parser.add_argument('--embedding-size', type=int, default=128, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--ini-channel', type=int, default=16, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--num-classes', type=int, default=1211, metavar='BS',
                    help='input batch size for training (default: 1211 5994)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.125)')
parser.add_argument('--lr-decay', default=1e-5, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')
parser.add_argument('--init-clip-max-norm', default=1.0, type=float,
                    metavar='CLIP', help='grad clip max norm (default: 1.0)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=2, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=100, metavar='LI',
                    help='how many batches to wait before logging training status')

#cut mix
parser.add_argument('--beta', default=0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float,
                    help='cutmix probability')
parser.add_argument('--is-training', type=int,
                    help='start from MFB file')
parser.add_argument('--pool-size', type=int, default=5000, metavar='LI',
                    help='pool size')
parser.add_argument('--baseWidth', type=int, default=7, metavar='LI',
                    help='baseWidth')
parser.add_argument('--scale', type=int, default=4, metavar='LI',
                    help='scale')                    
args = parser.parse_args()

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)


args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if args.cuda:
    cudnn.benchmark = False
torch.backends.cudnn.enabled = True

LOG_DIR = args.log_dir
os.system('chmod -R 777 %s'%LOG_DIR)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count

def myaccuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def main():
    #model = ResNetTimePooling_(args.embedding_size, args.num_classes, args.ini_channel)
    model = Res2Net(args.embedding_size, args.num_classes, args.ini_channel, args.baseWidth, args.scale)
    model.cuda()
    model = torch.nn.DataParallel(model)
    
    print('\nparsed options:\n{}\n'.format(vars(args)))
    print(LOG_DIR)
    loss_mean = []
    optimizer = create_optimizer(model, args.lr)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer.param_groups[0]['lr']=args.lr
            #loss_mean=np.load(LOG_DIR+'/loss_%s.npy'%(args.start_epoch-1)).tolist()
            loss_mean = []
            """
            tmp=[]
            for line in open(LOG_DIR+'/loss.txt'):
                if int(line.split(':')[1].split()[0])<args.start_epoch:
                    tmp.append(line)
            fp=open(LOG_DIR+'/loss.txt', 'w')
            for i in range(len(tmp)):
                fp.write(tmp[i])
            fp.close()
            """
        else:
            loss_mean=[]
            print('=> no checkpoint found at {}'.format(args.resume))
    #'''
    #load test data
    print("Read valid data...")
    dev_loader = TranLoader('/data/xiaorunqiu/data/vox2_fbank/vox1test_kaldi/feats.scp', pool_size=4000, batch_size=1, shuffle=True, feat_dim=64)
    valid = {}
    while(True):
        utt_batch, mat = dev_loader.generateBatch()
        if utt_batch is None:
            break
        mat = np.reshape(mat, [1, 1, mat.shape[1], mat.shape[2]])
        valid[utt_batch[0]] = torch.from_numpy(mat).float()
    print("Read valid data done.")
    #'''
    #train
    start = args.start_epoch
    end = start + args.epochs

    if args.is_training == 0:
        Valid(valid, model, start)
    else:
        #read train label
        label_train={}
        for line in open('list/train_npy.list'):
            line=line.strip().split(' ')
            utt=line[-1].split('/')[-1].split('.')[0]
            label_train[utt]= int(line[1])
        #TranLoader
        train_loader = TranLoader('/data/xiaorunqiu/data/vox2_fbank/vox2_kaldi/feats_all_0.scp', pool_size=args.pool_size, batch_size=args.batch_size, shuffle=True, feat_dim=64)
        for epoch in range(start, end):
            train(model, optimizer, train_loader, label_train, epoch, loss_mean)
            Valid(valid, model, epoch)


def train(model, optimizer, train_loader, label_train, epoch, loss_mean):
    # switch to train mode
    model.train()
    batch_size = args.batch_size
    criterion = nn.CrossEntropyLoss().cuda()
    top1       = AverageMeter() 
    top5       = AverageMeter()

    epoch_itr = int(train_loader.__len__() / batch_size)
    loss_tmp = 0.0
    itr = 0
    min_chunksize = 200 + 100*int(np.log10(0.1/args.lr))
    max_chunksize =400 + 100*int(np.log10(0.1/args.lr))
    print("Train chunk size:%s %s"%(min_chunksize, max_chunksize))
    while(True):
        t1=time.time()
        utt_batch, mat_batch = train_loader.generateBatch(chunk_size = np.random.randint(min_chunksize, max_chunksize))
        #utt_batch, mat_batch = train_loader.generateBatch(chunk_size = min_chunksize)
        if utt_batch is None:
            break
        t2=time.time()
        label_batch = np.array([label_train[utt_batch[i]] for i in range(batch_size)])
        mat_batch = np.reshape(mat_batch, [batch_size, 1, mat_batch.shape[1], mat_batch.shape[2]])
        data = torch.from_numpy(mat_batch).float().cuda()
        label = torch.from_numpy(label_batch).long().cuda()
        
        # compute output
        input_var = torch.autograd.Variable(data, requires_grad=True)
        label_var = Variable(label)
        
        out_feature, out_cls = model(input_var)
        loss = criterion(out_cls, label_var)
            
        prec1, prec5 = myaccuracy(out_cls.data, label.cuda(), topk=(1, 5))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))
        optimizer.zero_grad()
        loss.backward()
        loss_tmp += loss.item()

        utils.clip_grad_value_(model.parameters(), args.init_clip_max_norm)
        optimizer.step()
        t3 = time.time()
        itr += 1
        if itr % args.log_interval == 0 and itr!=0:
            loss_tmp /= args.log_interval
            fp = open(LOG_DIR+'/loss.txt','a')
            fp.write('Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tLoss: {:.6f}\tacc1: {:.6f}\tlr:{:.4f}\n'.format(
                epoch, itr, epoch_itr,
                100. * itr / epoch_itr,loss_tmp,
                top1.avg, optimizer.param_groups[0]['lr']))
            fp.close()
            loss_mean.append(loss_tmp)
            loss_tmp = 0.0
            '''
            #plot
            plot.figure()
            plot.plot(range(0,args.log_interval*len(loss_mean),args.log_interval), loss_mean)
            plot.vlines(range(0, args.log_interval*len(loss_mean), int(epoch_itr/args.log_interval)*args.log_interval), 0, 8)
            plot.title("loss")
            plot.xlabel("itr")
            plot.ylabel("loss")
            plot.savefig(LOG_DIR+'/loss.png', format='png', dpi=119)
            #plot
            plot.figure()
            if epoch > 10: 
                tmp=loss_mean[(epoch-10)*int(epoch_itr/args.log_interval):-1]
            else:
                tmp = loss_mean
            plot.plot(range(0,args.log_interval*len(tmp), args.log_interval), tmp)
            plot.vlines(range(0, args.log_interval*len(tmp), int(epoch_itr/args.log_interval)*args.log_interval), 0, max(tmp))
            plot.title("loss")
            plot.xlabel("itr")
            plot.ylabel("loss")
            plot.savefig(LOG_DIR+'/loss1.png', format='png', dpi=119)
            '''
            top1 = AverageMeter() 
            top5 = AverageMeter()
    # do checkpointing
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))
    np.save(LOG_DIR+'loss_%s'%epoch, np.array(loss_mean))

def Valid(valid, model, epoch):
    model.eval()    
    uttrance_feature={}
    label = torch.from_numpy(np.zeros(args.batch_size)).long()
    label = label.cuda() 
    label_var = Variable(label)
    for key in valid: 
        with torch.no_grad():
            if args.cuda:
                data = valid[key].cuda()
            data = Variable(data) 
            uttrance_feature[key] = model(data)[0][0].cpu()
            #output = model(data)
            
    score_gen=[]
    score_spo=[]
    fp = open('score_ce.txt', 'w')
    for line in open('list/veri_test.txt'):
        line=line.strip().split(' ')
        label = int(line[0]) 
        line[1] = '-'.join(line[1].split('/')).split('.')[0]
        line[2] = '-'.join(line[2].split('/')).split('.')[0]
        score = torch.nn.functional.cosine_similarity(uttrance_feature[line[1]], uttrance_feature[line[2]], dim=0).item()
        if label==1:
            score_gen.append(score)
        else:
            score_spo.append(score)
        fp.write('%s %s %s\n'%(line[1], line[2], score))
    fp.close()
    eer, _ = EER(score_gen, score_spo)
    eer_kaldi, mindcf1, mindcf2 = EER_kaldi('score_ce.txt', 'list/target.txt')
    #print("Epoch:%s EER:%.2f Positive:%s Negative:%s"%(epoch, eer, len(score_gen), len(score_spo)))
    fp = open(LOG_DIR+'/log.txt','a')
    fp.write('epoch: {}.\tEER:{:.3f} mindcf(0.01):{:.3f} mindcf(0.001):{:.3f}\n'.format(epoch, eer, mindcf1, mindcf2))
    fp.close()
    
def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    #optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
    #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=40, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    
    return optimizer

if __name__ == '__main__':
    main()
