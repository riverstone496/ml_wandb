import wandb
import matplotlib.pyplot as plt
import numpy as np

col = {'AllSGD':'tab:pink','SGDNorm':'tab:cyan','SGDDirection':'tab:green','None':'tab:red'}
model_name_dic = {'mlp':'MLP','cnn':'CNN','resnet18':'Resnet18','vit_tiny_patch16_224':'ViT-tiny'}
optim_dict = {'sgd':'SGD','adamw':'AdamW','psgd':'PSGD(KF)','kfac_mc':'K-FAC(global)','foof':'FOOF','skfac_mc':'SK-FAC(1-mc)','shampoo':'Shampoo','seng':'SENG','smw_ngd':'SMW-NG'}
grafting_dict = {'None':"solid",'SGDNorm':"dashdot",'SGDDirection':"dashed",'AllSGD':"solid"}

def make_dict(epoch_list):
    acc_dic={}
    for optim in optim_list:
        acc_dic[optim]={}
        for grafting in ["None",'SGDNorm','SGDDirection','AllSGD']:
            acc_dic[optim][grafting]={}
            for epoch in epoch_list:
                acc_dic[optim][grafting][epoch]=np.inf
    return acc_dic

def remove_dict(acc_dic,epoch_list):
    for optim in optim_list:
        for epoch in epoch_list:
            for grafting in ["None",'SGDNorm','SGDDirection','AllSGD']:
                if acc_dic[optim][grafting][epoch]==1:
                    acc_dic[optim][grafting].pop(epoch)
    return acc_dic

def collect_runs(model,bs,sweep_list_mlp,epoch_list,metric='max_test_accuracy'):
    damp_loss_dic = make_dict(epoch_list)
    for sweep_name in sweep_list_mlp:
        sweep = api.sweep(sweep_name)
        runs = sweep.runs
        for run in runs:
            model_name = run.config.get('model')
            if model_name == model  and run.config.get('epochs') in epoch_list and run.config.get('optim') in optim_list and run.config.get('batch_size') == bs:
                if run.summary.get("cuda_max_memory") == -1 or type(run.summary.get(metric)) != float:
                    test_accuracy=0
                else:
                    test_accuracy=run.summary.get(metric)/100

                cur_loss=damp_loss_dic[run.config.get('optim')][str(run.config.get('grafting'))][run.config.get('epochs')]
                if 1-test_accuracy < cur_loss:
                    damp_loss_dic[run.config.get('optim')][str(run.config.get('grafting'))][run.config.get('epochs')]=1-test_accuracy

                # if run.config.get('optim')=='kfac_mc' and run.config.get('ema_decay')==-1:
                #     cur_loss=damp_loss_dic['kfac_mc_local'][str(run.config.get('grafting'))][run.config.get('epochs')]
                #     if 1-test_accuracy < cur_loss:
                #         damp_loss_dic['kfac_mc_local'][str(run.config.get('grafting'))][run.config.get('epochs')]=1-test_accuracy
    print(damp_loss_dic)
    return damp_loss_dic

def plot(ax,output_dic,optim):
    for grafting in ["None",'SGDNorm','SGDDirection']:
        xkey=list(output_dic[optim][grafting].keys())
        yvalue=list(output_dic[optim][grafting].values())
        ax.plot(xkey,yvalue,label=grafting,color=col[grafting],linestyle = grafting_dict[grafting],marker='o')

    optim,grafting = 'sgd','AllSGD'
    xkey=list(output_dic[optim][grafting].keys())
    yvalue=list(output_dic[optim][grafting].values())
    ax.plot(xkey,yvalue,label=grafting,color=col[grafting],linestyle = grafting_dict[grafting],marker='o')

if __name__=='__main__':
    api = wandb.Api()
    optim_list = ['sgd','shampoo','psgd','kfac_mc','foof']
    epoch_list = [5,10,20,40]
    epoch_list_resnet = [25,50,100,200]
    filename = './graph/grafting.png'

    sweep_list_mlp={
        'riverstone/grafting/godore5g',
        'riverstone/grafting/yvvvpeqq',
        'riverstone/grafting/6uu1uny6',
        'riverstone/grafting/u7szv79y',
        'riverstone/grafting/gvi0d9s3',
        'riverstone/grafting/xr4k3q2n'
    }

    sweep_list_resnet={
        'riverstone/grafting/cf10xbtw',
        'riverstone/grafting/rgdrwvna',
        'riverstone/grafting/oriqygl6',
        'riverstone/grafting/s45umgc1'
    }

    sweep_list_vit={
        'riverstone/grafting/0fq4zsll',
        'riverstone/grafting/9t8riv33',
        'riverstone/grafting/l8skh2i3',
        'riverstone/grafting/x54o7iz8'
    }

    test_dic_mlp_256=collect_runs('mlp',256,sweep_list_mlp,epoch_list,metric='test_accuracy')
    test_dic_mlp=collect_runs('mlp',16384,sweep_list_mlp,epoch_list,metric='test_accuracy')
    test_dic_resnet=collect_runs('resnet18',128,sweep_list_resnet,epoch_list_resnet,metric='test_accuracy')
    test_dic_resnet_1024=collect_runs('resnet18',1024,sweep_list_resnet,epoch_list_resnet,metric='test_accuracy')
    test_dic_vit=collect_runs('vit_tiny_patch16_224',512,sweep_list_vit,epoch_list,metric='test_accuracy')
    test_dic_vit=remove_dict(test_dic_vit,epoch_list)

    plt.rcParams["font.size"] = 28
    fig, axes = plt.subplots(nrows=5, ncols=len(optim_list)-1, figsize=(40, 50))
    optim_list.remove('sgd')

    for j in range(len(optim_list)):
        optim = optim_list[j]
        plot(axes[0][j],test_dic_mlp_256,optim)
        plot(axes[1][j],test_dic_mlp,optim)
        plot(axes[2][j],test_dic_resnet,optim)
        plot(axes[3][j],test_dic_resnet_1024,optim)
        plot(axes[4][j],test_dic_vit,optim)
        
        axes[0][j].set_yscale('log',base=2)
        axes[1][j].set_yscale('log',base=2)
        axes[2][j].set_yscale('log',base=2)
        axes[3][j].set_yscale('log',base=2)
        axes[4][j].set_yscale('log',base=2)

        axes[0][j].set_xlabel('Epoch')
        axes[1][j].set_xlabel('Epoch')
        axes[2][j].set_xlabel('Epoch')
        axes[3][j].set_xlabel('Epoch')
        axes[0][0].set_ylabel('MLP(bs=256)')
        axes[1][0].set_ylabel('MLP(bs=16384)')
        axes[2][0].set_ylabel('Resnet18(bs=128)')
        axes[3][0].set_ylabel('Resnet18(bs=1024)')
        axes[4][0].set_ylabel('ViT-T')
        axes[0][j].set_title(optim)
    
    plt.legend(loc='upper center', bbox_to_anchor=(0, -0.2), ncol=4)
    plt.plot()
    plt.savefig(filename,dpi=80,bbox_inches='tight',pad_inches=0.1)
