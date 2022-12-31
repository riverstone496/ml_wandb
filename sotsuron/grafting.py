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
    filename = './sotsuron/graph/grafting.png'

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

    metric = "max_test_accuracy"
    test_dic_mlp_256=collect_runs('mlp',256,sweep_list_mlp,epoch_list,metric=metric)
    test_dic_mlp=collect_runs('mlp',16384,sweep_list_mlp,epoch_list,metric=metric)
    test_dic_resnet=collect_runs('resnet18',128,sweep_list_resnet,epoch_list_resnet,metric=metric)
    test_dic_resnet_1024=collect_runs('resnet18',1024,sweep_list_resnet,epoch_list_resnet,metric=metric)
    test_dic_vit=collect_runs('vit_tiny_patch16_224',512,sweep_list_vit,epoch_list,metric=metric)
    test_dic_resnet=remove_dict(test_dic_resnet,epoch_list_resnet)
    test_dic_resnet_1024=remove_dict(test_dic_resnet_1024,epoch_list_resnet)
    test_dic_vit=remove_dict(test_dic_vit,epoch_list)

    plt.rcParams["font.size"] = 28
    fig, axes = plt.subplots(nrows=len(graft_list), ncols=5, figsize=(50, 30))

    for j in range(len(graft_list)):
        graft = graft_list[j]
        plot(axes[j][0],test_dic_mlp_256,graft)
        plot(axes[j][1],test_dic_mlp,graft)
        plot(axes[j][2],test_dic_resnet,graft)
        plot(axes[j][3],test_dic_resnet_1024,graft)
        plot(axes[j][4],test_dic_vit,graft)

        axes[j][0].set_xlabel('Epoch')
        axes[j][1].set_xlabel('Epoch')
        axes[j][2].set_xlabel('Epoch')
        axes[j][3].set_xlabel('Epoch')
        axes[j][4].set_xlabel('Epoch')
        axes[j][0].set_ylabel(graft)

        axes[j][0].set_title('MLP(bs=128)')
        axes[j][1].set_title('MLP(bs=16384)')
        axes[j][2].set_title('Resnet(bs=128)')
        axes[j][3].set_title('Resnet(bs=1024)')
        axes[j][4].set_title('ViT')

    plt.legend(loc='upper center', bbox_to_anchor=(0, -0.2), ncol=3)
    plt.plot()
    plt.savefig(filename,dpi=80,bbox_inches='tight',pad_inches=0.1)
