from utility.helper import *
from utility.batch_test import test
from utility.parser import parse_args
from utility.load_data import *
from AbtCDR import AbtCDR
import scipy.sparse as sp
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from random import randint,random
from tqdm import tqdm
import numpy as np
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()

def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data

def get_adj_mat(config,data_generator,adj_type,domain_type):
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    if adj_type == 'plain':
        config['norm_adj_%s'%domain_type] = plain_adj
        print('%s use the plain adjacency matrix'%domain_type)

    elif adj_type == 'norm':
        config['norm_adj_%s'%domain_type] = norm_adj
        print('%s use the normalized adjacency matrix'%domain_type)

    elif adj_type == 'gcmc':
        config['norm_adj_%s'%domain_type] = mean_adj
        print('%s use the gcmc adjacency matrix'%domain_type)

    else:
        config['norm_adj_%s'%domain_type] = mean_adj + sp.eye(mean_adj.shape[0])
        print('%s use the mean adjacency matrix'%domain_type)

def print_test_result(hr,ndcg,train_time,test_time,domain_type,data_status):
    if args.verbose > 0:
        perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.4f=%.4f + %.4f], hit=%s, ndcg=%s at %s' % \
                (epoch, train_time, test_time, losses, loss_source,loss_target , str(['%.4f'%i for i in hr]),
                    str(['%.4f'%i for i in ndcg]),data_status)
        pprint(perf_str,save_log_file)

def find_best_epoch(ndcg_loger,hit_loger,domain_type):
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)
    hit_10 = hit[:,-1]
    best_rec_0 = max(hit_10)
    idx = list(hit_10).index(best_rec_0)
    pprint('{:*^40}'.format(domain_type+' part'),save_log_file)
    final_perf = "Best Iter=[%d]@[%.1f]\t hit=%s, ndcg=%s" % \
                 (idx-1, time() - t0, str(['%.4f'%i for i in list(hit[idx])]), str(['%.4f'%i for i in list(ndcgs[idx])]))
    pprint(final_perf,save_log_file)
    return final_perf

if __name__ == '__main__':
    args = parse_args()
    global best_hr_s
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    ## 1 ->2, 4->5, 0->0
    
    # torch.cuda.set_device(2)
    save_path = '%soutput/%s.result' % (args.proj_path, args.dataset)
    
    weight_id = random()
    save_log_dir = './logs/jujingxin/%s/%s/'%(args.dataset,str(args.layer_size))
    ensureDir(save_log_dir)
    save_log_file = open(save_log_dir+'lr_%s_b%s_id_%.4f.txt'%(str(args.lr),args.batch_size,weight_id),'w+')
    config = dict()
    Ks = eval(args.Ks)
    layer_size = args.layer_size
    BATCH_SIZE = args.batch_size
    neg_num = args.neg_num
    source_name,target_name = args.dataset.split('_')
    data_generator_s = Data(path=args.data_path + args.dataset, batch_size=args.batch_size,neg_num=neg_num)
    data_generator_t = Data(path=args.data_path + target_name+'_'+source_name, batch_size=args.batch_size,neg_num=neg_num)
    pprint(config, save_log_file)
    pprint(args,save_log_file)
    pprint('{:*^40}'.format('source data info' ),save_log_file)
    data_generator_s.print_statistics(save_log_file)
    pprint('{:*^40}'.format('target data info' ),save_log_file)
    data_generator_t.print_statistics(save_log_file)
    assert data_generator_s.n_users == data_generator_t.n_users ,'data-erro,user should be shared'

    '''compute domain_adj '''
    domain_adj = sp.dok_matrix((data_generator_s.n_users,2),dtype=np.float32)
    domain_adj = domain_adj.tolil()
    R_s = data_generator_s.get_R_mat()
    R_t = data_generator_t.get_R_mat()
    domain_adj[:,0] = R_s.sum(1)
    domain_adj[:,1] = R_t.sum(1)
    domain_adj = domain_adj.todok()
    degree_sum = np.array(domain_adj.sum(1))
    d_inv = np.power(degree_sum, -1)
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv[:,0])  
    norm_domain_adj = d_mat_inv.dot(domain_adj)
    config['domain_adj'] = np.array(norm_domain_adj.todense())


    config['n_users'] = data_generator_s.n_users
    config['n_items_s'] = data_generator_s.n_items
    config['n_items_t'] = data_generator_t.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    get_adj_mat(config,data_generator_s,args.adj_type,'s')
    get_adj_mat(config,data_generator_t,args.adj_type,'t')

    t0 = time()

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

        #get sparcy station
    if args.sparcy_flag:
        split_ids_s,split_status_s = data_generator_s.get_sparsity_split()
        split_ids_t,split_status_t = data_generator_t.get_sparsity_split()

    else:
        split_ids_s,split_ids_t,split_status_s,split_status_t=[],[],[],[]
    split_ids_s.append(range(data_generator_s.n_users))
    split_ids_t.append(range(data_generator_t.n_users))
    split_status_s.append('full rating, #user=%d'%data_generator_s.n_users)
    split_status_t.append('full rating, #user=%d'%data_generator_t.n_users)

    model = AbtCDR(config=config, args=args)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print("----")

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    pre_loger_t, rec_loger_t, ndcg_loger_t, hit_loger_t = [], [], [], []

    """
    *********************************************************
    Train.
    """
    stopping_step = 0
    stopping_step_s = 0
    should_stop_s = False
    should_stop_t = False
    verbose = 10
    isonebatch = False
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_source = optimizer
    optimizer_target = optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_source = torch.optim.Adam([model.source_user_embedding, model.source_item_embedding], lr=args.lr)
    optimizer_target = torch.optim.Adam([model.target_user_embedding, model.target_item_embedding], lr=args.lr) 
    for epoch in range(args.epoch):
        model.train()
        optimizer_source.zero_grad()
        t1 = time()
        loss, loss_source, loss_target, single_loss_source, single_loss_target = [],[],[],[],[]
        user_input_s,item_input_s,label_s = data_generator_s.get_train_instance()
        user_input_t,item_input_t,label_t = data_generator_t.get_train_instance()
        train_len_s = len(user_input_s)
        train_len_t = len(user_input_t)
        shuffled_idx_s = np.random.permutation(np.arange(train_len_s))
        train_u_s = user_input_s[shuffled_idx_s]
        train_i_s = item_input_s[shuffled_idx_s]
        train_r_s = label_s[shuffled_idx_s]
        shuffled_idx_t = np.random.permutation(np.arange(train_len_t))
        train_u_t = user_input_t[shuffled_idx_t]
        train_i_t = item_input_t[shuffled_idx_t]
        train_r_t = label_t[shuffled_idx_t]
        n_batch_s = train_len_s // args.batch_size + 1
        n_batch_t = train_len_t // args.batch_size + 1
        n_batch_max = max(n_batch_s,n_batch_t)
        n_batch_min = min(n_batch_s,n_batch_t)


        model.train()
        for name, param in model.named_parameters():
            param.requires_grad = True
        for i in tqdm(range(n_batch_min),desc='train_join',ascii=True):
            optimizer.zero_grad()
            min_idx = i*BATCH_SIZE
            max_idx = np.min([(i+1)*BATCH_SIZE,min([train_len_s,train_len_t])])
            if max_idx<(i+1)*BATCH_SIZE:
                idex = list(range(min_idx,max_idx))+list(np.random.randint(0,min([train_len_s,train_len_t]),(i+1)*BATCH_SIZE-max_idx))
                train_u_batch_s = train_u_s[idex]
                train_i_batch_s = train_i_s[idex]
                train_r_batch_s = train_r_s[idex]                
                train_u_batch_t = train_u_t[idex]
                train_i_batch_t = train_i_t[idex]
                train_r_batch_t = train_r_t[idex]
            else:
                train_u_batch_s = train_u_s[min_idx: max_idx]
                train_i_batch_s = train_i_s[min_idx: max_idx]
                train_r_batch_s = train_r_s[min_idx: max_idx]
                train_u_batch_t = train_u_t[min_idx: max_idx]
                train_i_batch_t = train_i_t[min_idx: max_idx]
                train_r_batch_t = train_r_t[min_idx: max_idx]
            loss_source, loss_target, losses = model.calculate_cross_loss(train_u_batch_s, train_i_batch_s, train_r_batch_s, train_u_batch_t, train_i_batch_t, train_r_batch_t)
            loss.append(losses)
            single_loss_source.append(loss_source)
            single_loss_target.append(loss_target)
        
            losses.backward()
            optimizer.step()
        
        if n_batch_s>=n_batch_t:

            pprint('source domain single train',save_log_file)
            
            for name, param in model.named_parameters():
                    if "target" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            for i in tqdm(range(n_batch_min,n_batch_max),desc='train_source',ascii=True):
                optimizer_source.zero_grad()
                min_idx = i*BATCH_SIZE
                max_idx = np.min([(i+1)*BATCH_SIZE,train_len_s])
                if max_idx<(i+1)*BATCH_SIZE:
                    idex = list(range(min_idx,max_idx))+list(np.random.randint(0,train_len_s,(i+1)*BATCH_SIZE-max_idx))
                    train_u_batch = train_u_s[idex]
                    train_i_batch = train_i_s[idex]
                    train_r_batch = train_r_s[idex]
                else:
                    train_u_batch = train_u_s[min_idx: max_idx]
                    train_i_batch = train_i_s[min_idx: max_idx]
                    train_r_batch = train_r_s[min_idx: max_idx]

                batch_loss_source, _ = model.calculate_single_loss(train_u_batch, train_i_batch, train_r_batch, "source")
                single_loss_source.append(batch_loss_source)

                
                batch_loss_source.backward()
                optimizer_source.step()
                
            
        else:
            pprint('target domain single train',save_log_file)
            for name, param in model.named_parameters():
                if "source" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            for i in tqdm(range(n_batch_min,n_batch_max),desc='train_target',ascii=True):
               
                optimizer_target.zero_grad()
                min_idx = i*BATCH_SIZE
                max_idx = np.min([(i+1)*BATCH_SIZE,train_len_t])
                if max_idx<(i+1)*BATCH_SIZE:
                    idex = list(range(min_idx,max_idx))+list(np.random.randint(0,train_len_t,(i+1)*BATCH_SIZE-max_idx))
                    train_u_batch = train_u_t[idex]
                    train_i_batch = train_i_t[idex]
                    train_r_batch = train_r_t[idex]
                else:
                    train_u_batch = train_u_t[min_idx: max_idx]
                    train_i_batch = train_i_t[min_idx: max_idx]
                    train_r_batch = train_r_t[min_idx: max_idx]
                _, batch_loss_target= model.calculate_single_loss(train_u_batch, train_i_batch, train_r_batch, "target")
                
                single_loss_target.append(batch_loss_target)
                
                batch_loss_target.backward()
                optimizer_target.step()

        loss_target_mean = torch.mean(torch.tensor(single_loss_target))
        loss_source_mean = torch.mean(torch.tensor(single_loss_source))
        loss_total = torch.mean(torch.tensor(loss))
        print(f"epoch:{epoch},loss_source:{loss_source_mean}, loss_target:{loss_target_mean}, loss_total:{loss_total}")

        model.eval()
        t2 = time()
        #------------------batch_test.py---------------
        pprint('\n' + '{:*^40}'.format('source result'), save_log_file)
        for test_user_list_s,data_status in zip(split_ids_s,split_status_s):
            hr_s,ndcg_s = test(model, data_generator_s,test_user_list_s,'source',2048, Ks,layer_size)
            print_test_result(hr_s,ndcg_s,t2-t1,time()-t2,'source',data_status)
        t3 = time()
        pprint('\n'+'{:*^40}'.format('target result'),save_log_file)
        for test_user_list_t,data_status in zip(split_ids_t,split_status_t):
            hr_t,ndcg_t = test(model, data_generator_t,test_user_list_t,'target', 2048,Ks, layer_size)
            print_test_result(hr_t,ndcg_t,t2-t1,time()-t3,'target',data_status)
        t4 = time()

        
        loss_loger.append(loss)
        ndcg_loger.append(ndcg_s)
        hit_loger.append(hr_s)

        ndcg_loger_t.append(ndcg_t)
        hit_loger_t.append(hr_t)

        if epoch == 0:
            best_hr_t = hr_t[-1]
            best_hr_s = hr_s[-1]
        else:
            best_hr_s, stopping_step_s, should_stop_s = early_stopping(hr_s[-1], best_hr_s, stopping_step_s, flag_step=5)
            best_hr_t,stopping_step, should_stop_t = early_stopping(hr_t[-1], best_hr_t, stopping_step, flag_step=5)


        if all([should_stop_s,should_stop_t]) == True:
            break 

    final_perf_s = find_best_epoch(ndcg_loger,hit_loger,'source')
    final_perf_t = find_best_epoch(ndcg_loger_t,hit_loger_t,'target')
    save_path = '%soutput/%s.result' % (args.proj_path, args.dataset)
    ensureDir(save_path)
    

    f = open(save_path, 'a')

    f.write(
        '\n lr=%.4f, layer_fun=%s,fuse_type_in =%s,neg_num=%s, n_interaction=%s,connect_type=%s\n\t%s\n%s\n%s\n%s'
        % (args.lr, args.layer_fun, args.fuse_type_in,args.neg_num,
           args.n_interaction,args.connect_type, 'source result',final_perf_s,'target result',final_perf_t))
    f.close()
