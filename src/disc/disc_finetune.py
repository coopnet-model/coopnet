import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils import shuffle
from datasets import adj_sent
from transformers import *
from utils import (encode_dataset_bert_sent, iter_data, ResultLogger, make_path)
import pickle
from tensorboard_logger import configure, log_value

configure("./disc_analysis", flush_secs=5)
def transform_ab(X1,X2):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, 500), dtype=np.int32)
    mmb = np.zeros((n_batch, 500), dtype=np.float32)
    for i, (x1,x2), in enumerate(zip(X1,X2)):
        new_x1 = x1[:-1] + [102, x2[1]] + [x1[-1]]
        xmb[i,:len(new_x1)] = new_x1
        mmb[i,:len(new_x1)] = 1
    return xmb, mmb

def iter_apply(Xs, Ys):
    logits = []
    losses = []
    with torch.no_grad():
        dh_model.eval()
        for xmb, ymb in iter_data(Xs, Ys, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            YMB = torch.tensor(ymb, dtype=torch.long).to(device)
            outputs = dh_model(XMB,next_sentence_label=YMB)
            loss, scores = outputs[:2]
            losses.append(float(loss))
    return np.sum(losses), np.mean(losses)

def log(save_dir, desc,iter=0,save=''):
    global best_score
    print("Logging")
    tr_sum_loss, tr_mean_loss = iter_apply(trX[:n_valid], trY[:n_valid])
    va_sum_loss, va_mean_loss = iter_apply(vaX[:n_valid], vaY[:n_valid])
    log_value('va_sum_loss',va_sum_loss,n_updates)
    log_value('va_mean_loss',va_mean_loss,n_updates)
    log_value('tr_mean_loss',tr_mean_loss,n_updates)
    log_value('tr_sum_loss',tr_sum_loss,n_updates)
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=float(tr_mean_loss), va_cost=float(va_mean_loss), tr_acc=float(tr_sum_loss), va_acc=float(va_sum_loss))
    print('%d %d %.3f %.3f %.2f %.2f' % (n_epochs, n_updates, tr_sum_loss, va_sum_loss, tr_mean_loss, va_mean_loss))
    path = os.path.join(save_dir, desc, 'best_params_' + str(iter) + save)
    torch.save(dh_model.state_dict(), make_path(path))

def run_epoch(iter):
    losses = []
    i = 0
    for xmb,ymb in iter_data(*shuffle(trX, trY, random_state=np.random),
                                   n_batch=n_batch_train, truncate=True, verbose=True):
        global n_updates
        dh_model.train()
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        YMB = torch.tensor(ymb, dtype=torch.long).to(device)
        scores = dh_model(XMB)
        scores = scores[0]
        loss = loss_function(scores.view(-1,2),YMB.view(-1))
        losses.append(float(loss))
        loss.backward()
        model_opt.step()
        model_opt.zero_grad()
        n_updates += 1
        if (n_updates + 1) % 10000 == 0:
           print(acc)
        if (n_updates + 1) % 10000 == 0:
           log(save_dir, desc,iter,save='_'+str(n_updates))

        log_value('batch_train_loss',loss,n_updates)
        log_value('mean_train_loss',np.mean(losses),n_updates)
        log_value('total_train_loss',np.sum(losses),n_updates)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, default='model',help="Description")
    parser.add_argument('--dataset', type=str,default='bio')
    parser.add_argument('--common_match',type=str,default='physics')
    parser.add_argument('--log_dir', type=str, default='./log/')
    parser.add_argument('--save_dir', type=str, default='./model')
    parser.add_argument('--data_dir', type=str, default='../../data/finished_files_cat')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=20)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.00002)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Constants
    dataset = args.dataset
    n_ctx = args.n_ctx
    save_dir = args.save_dir
    desc = args.desc
    data_dir = args.data_dir
    log_dir = args.log_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
    disc_encoder = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')
    n_vocab = len(disc_encoder)
    print("Encoding dataset...")
    cat = args.dataset
    try:
        data_dump = pickle.load(open(args.data_dir + '/disc_' + args.dataset + '_sent.pkl','rb'))
        trX1, trX2, trY, trIds = data_dump[0]
        vaX1, vaX2, vaY, vaIds = data_dump[1]
        teX1, teX2, teY, teIds = data_dump[2]
    except:
        ((trX1, trX2, trY, trIds),
         (vaX1, vaX2, vaY, vaIds),
         (teX1, teX2, teY, teIds)) = encode_dataset_bert_sent(*adj_sent(data_dir,use_cat=True, cat=cat,not_cat=args.common_match),encoder=disc_encoder)
        pickle.dump([(trX1,trX2, trY, trIds), (vaX1, vaX2,vaY, vaIds), (teX1, teX2, teY, teIds)], open(args.data_dir + '/disc_' + args.dataset + '_sent.pkl','wb'))
    
    try:
       trX, trM, vaX, vaM =  pickle.load(open(args.data_dir + '/t_disc_' + args.dataset + '_sent.pkl','rb'))
    except:
       trX, trM = transform_ab(trX1, trX2)
       vaX, vaM = transform_ab(vaX1, vaX2)
       pickle.dump((trX,trM,trY, vaX,vaM,vaY), open(args.data_dir + '/t_disc_' + args.dataset + '_sent.pkl','wb'))

    n_train = len(trX)
    n_valid = len(vaX)
    n_batch_train = args.n_batch * max(n_gpu, 1)
    dh_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    dh_model.bert = AutoModel.from_pretrained('allenai/scibert_scivocab_cased')
    dh_model = dh_model.to(device)
    criterion = nn.CrossEntropyLoss(reduce=False)
    print(args.lr)
    model_opt = BertAdam(dh_model.parameters(),
                           lr=args.lr)
    loss_function = nn.CrossEntropyLoss()


    n_updates = 0
    n_epochs = 0

    best_score = 0
    for i in range(args.n_iter):
        print("running epoch", i)
        run_epoch(n_epochs)
        n_epochs += 1
        log(save_dir, desc,i)
