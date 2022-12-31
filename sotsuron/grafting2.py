import wandb
import matplotlib.pyplot as plt
import numpy as np

col = {'sgd':'tab:pink','psgd':'tab:red','kfac_mc':'tab:green','kfac_mc_local':'tab:cyan','seng':'tab:blue','shampoo':'tab:purple','adamw':'tab:brown','foof':'tab:olive'}
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
                acc_dic[optim][grafting][epoch]=0
    return acc_dic

def remove_dict(acc_dic,epoch_list):
    for optim in optim_list:
        for epoch in epoch_list:
            for grafting in ["None",'SGDNorm','SGDDirection','AllSGD']:
                if acc_dic[optim][grafting][epoch]<50:
                    acc_dic[optim][grafting].pop(epoch)
    return acc_dic

def collect_runs(model,bs,sweep_list_mlp,epoch_list,metric='max_test_accuracy'):
    damp_loss_dic = make_dict(epoch_list)
    for sweep_name in sweep_list_mlp:
        sweep = api.sweep(sweep_name)
        runs = sweep.runs
        for run in runs:
            model_name = run.config.get('model')
            opt=str(run.config.get('optim'))
            graft=str(run.config.get('grafting'))

            if model_name == model  and run.config.get('epochs') in epoch_list and opt in optim_list and run.config.get('batch_size') == bs:
                if run.summary.get("cuda_max_memory") == -1 or type(run.summary.get(metric)) != float:
                    test_accuracy=0
                else:
                    test_accuracy=run.summary.get(metric)

                if graft == 'AllSGD':
                    opt = 'sgd'

                cur_loss=damp_loss_dic[opt][graft][run.config.get('epochs')]
                if test_accuracy > cur_loss:
                    damp_loss_dic[opt][graft][run.config.get('epochs')]=test_accuracy

                # if run.config.get('optim')=='kfac_mc' and run.config.get('ema_decay')==-1:
                #     cur_loss=damp_loss_dic['kfac_mc_local'][str(run.config.get('grafting'))][run.config.get('epochs')]
                #     if test_accuracy > cur_loss:
                #         damp_loss_dic['kfac_mc_local'][str(run.config.get('grafting'))][run.config.get('epochs')]=test_accuracy
    print(model,' completed')
    return damp_loss_dic

def plot(ax,output_dic,grafting):
    for optim in optim_list: 
        if optim == 'sgd':
            xkey=list(output_dic[optim]['AllSGD'].keys())
            yvalue=list(output_dic[optim]['AllSGD'].values())
            ax.plot(xkey,yvalue,label=optim_dict[optim],color=col[optim],marker='o')
        else:
            xkey=list(output_dic[optim][grafting].keys())
            yvalue=list(output_dic[optim][grafting].values())
            ax.plot(xkey,yvalue,label=optim_dict[optim],color=col[optim],marker='o')

if __name__=='__main__':
    api = wandb.Api()
    optim_list = ['sgd','shampoo','psgd','kfac_mc','foof','seng']
    graft_list = ["None",'SGDNorm','SGDDirection']
    epoch_list = [5,10,20,40]
    epoch_list_resnet = [25,50,100,200]

    model = 'mlp'
    #model = 'vit_tiny_patch16_224'
    batch_size = 256
    filename = './sotsuron/graph/grafting/grafting_'+model+'_bs'+str(batch_size)+'.png'

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
        'riverstone/grafting/s45umgc1',
    }

    sweep_list_vit={
        'riverstone/grafting/0fq4zsll',
        'riverstone/grafting/9t8riv33',
        'riverstone/grafting/l8skh2i3',
        'riverstone/grafting/x54o7iz8'
    }

    if model == 'mlp':
        sweep_list=sweep_list_mlp
        epoch_list=epoch_list
    if model == 'resnet18':
        sweep_list=sweep_list_resnet
        epoch_list=epoch_list_resnet
    if model == 'vit_tiny_patch16_224':
        sweep_list=sweep_list_vit
        epoch_list=epoch_list

    metric = "max_test_accuracy"
    test_dic = collect_runs(model,batch_size,sweep_list,epoch_list,metric=metric)
    test_dic=remove_dict(test_dic,epoch_list)

    plt.rcParams["font.size"] = 28
    fig, axes = plt.subplots(nrows=1, ncols=len(graft_list), figsize=(30,10))

    for j in range(len(graft_list)):
        graft = graft_list[j]
        plot(axes[j],test_dic,graft)

        axes[j].set_xlabel('Epoch')
        axes[j].set_ylabel(graft)

    axes[1].set_title(model_name_dic[model]+'(bs='+str(batch_size)+')')
    plt.legend(loc='upper center', bbox_to_anchor=(0, -0.2), ncol=3)
    plt.plot()
    plt.savefig(filename,dpi=80,bbox_inches='tight',pad_inches=0.1)
