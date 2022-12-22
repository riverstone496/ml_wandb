import wandb
import matplotlib.pyplot as plt
import numpy as np

col = {'sgd':'tab:pink','psgd':'tab:red','kfac_mc':'tab:green','kfac_mc_local':'tab:cyan','seng':'tab:blue','shampoo':'tab:purple','adamw':'tab:brown','foof':'tab:olive'}
model_name_dic = {'mlp':'MLP','cnn':'CNN','resnet18':'Resnet18','vit_tiny_patch16_224':'ViT-tiny'}
optim_dict = {'sgd':'SGD','adamw':'AdamW','psgd':'PSGD(KF)','kfac_mc':'K-FAC(global)','kfac_mc_local':'K-FAC(local)','skfac_mc':'SK-FAC(1-mc)','shampoo':'Shampoo','seng':'SENG','smw_ngd':'SMW-NG','foof':'FOOF'}

def make_dict():
    acc_dic={}
    for optim in optim_list:
        acc_dic[optim]={}
        for bs in batchsize_list:
            acc_dic[optim][bs]=np.inf
    return acc_dic

def collect_runs(sweep_list,metric='test_accuracy'):
    damp_loss_dic = make_dict()
    for sweep_name in sweep_list:
        sweep = api.sweep(sweep_name)
        runs = sweep.runs
        for run in runs:
            model_name = run.config.get('model')
            if model_name == model  and run.config.get('batch_size') in batchsize_list and run.config.get('optim') in optim_list:
                if run.summary.get("cuda_max_memory") == -1 or type(run.summary.get(metric)) != float:
                    test_accuracy=0
                else:
                    test_accuracy=run.summary.get(metric)/100

                cur_loss=damp_loss_dic[run.config.get('optim')][run.config.get('batch_size')]
                if 1-test_accuracy < cur_loss:
                    damp_loss_dic[run.config.get('optim')][run.config.get('batch_size')]=1-test_accuracy

                if run.config.get('optim')=='kfac_mc' and run.config.get('ema_decay')==-1:
                    cur_loss=damp_loss_dic['kfac_mc_local'][run.config.get('batch_size')]
                    if 1-test_accuracy < cur_loss:
                        damp_loss_dic['kfac_mc_local'][run.config.get('batch_size')]=1-test_accuracy

    print(damp_loss_dic)
    return damp_loss_dic

if __name__=='__main__':
    api = wandb.Api()
    optim_list = ['sgd','adamw','shampoo','kfac_mc','kfac_mc_local','seng','psgd','foof']
    batchsize_list = [256,512,1024,2048,4096,8192,16384]
    model='mlp'
    filename = './graph/ipsj2.png'

    sweep_list={
        'riverstone/criteria/spzpahql',
        'riverstone/criteria/denix8xi',
        'riverstone/optcriteria/e11b9g2v',
        'riverstone/grad_maker/5sweoxrv',
        'riverstone/grad_maker/xb0y5kd5',
        'riverstone/grad_maker/n727u0s3',
        'riverstone/grad_maker/yqid66vz'
    }

    clip_train_dic=collect_runs(sweep_list,metric='train_accuracy')
    clip_test_dic=collect_runs(sweep_list,metric='test_accuracy')

    plt.rcParams["font.size"] = 28
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 21))

    for j in range(len(optim_list)):
        optim = optim_list[j]
        clip_dampli=list(clip_train_dic[optim].keys())
        clip_lossli=list(clip_train_dic[optim].values())
        axes[0].plot(clip_dampli,clip_lossli,label=optim_dict[optim],color=col[optim],marker='o')

        clip_dampli=list(clip_test_dic[optim].keys())
        clip_lossli=list(clip_test_dic[optim].values())
        axes[1].plot(clip_dampli,clip_lossli,label=optim_dict[optim],color=col[optim],marker='o')

        axes[0].set_xscale('log',base=2)
        axes[1].set_xscale('log',base=2)
        
        axes[0].set_yscale('log',base=2)
        axes[1].set_yscale('log',base=2)

        axes[0].set_xlabel('Batch size')
        axes[1].set_xlabel('Batch size')
        axes[0].set_ylabel('Train Error rate')
        axes[1].set_ylabel('Test Error rate')
        axes[0].set_title('MLP')
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.plot()
    plt.savefig(filename,dpi=80,bbox_inches='tight',pad_inches=0.1)
