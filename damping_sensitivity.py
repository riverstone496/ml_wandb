import wandb
import matplotlib.pyplot as plt
import numpy as np

col = {'psgd':'tab:red','kbfgs':'tab:brown','kfac_mc':'tab:green','seng':'tab:blue','shampoo':'tab:purple'}
lstyle = {32:'dashed',512:'solid'}

def make_dict():
    acc_dic={}
    for model in  model_list:
        acc_dic[model] = {}
        for optim in optim_list:
            acc_dic[model][optim]={}
            for bs in batchsize_list:
                acc_dic[model][optim][bs]={}
                for damping in damping_list:
                    acc_dic[model][optim][bs][damping]=np.inf
    return acc_dic

def collect_runs(sweep_list):
    damp_loss_dic = make_dict()
    for sweep_name in sweep_list:
        sweep = api.sweep(sweep_name)
        runs = sweep.runs
        for run in runs:
            if run.config.get('model') == 'mlp':
                model_name = 'mlp_'+str(run.config.get('width'))
            else:
                model_name = run.config.get('model')

            if model_name in model_list  and run.config.get('batch_size') in batchsize_list and run.config.get('optim') in optim_list:
                if run.summary.get("cuda_max_memory") == -1 or type(run.summary.get("test_loss")) != float:
                    test_loss=np.inf
                else:
                    test_loss=run.summary.get("test_loss")
                cur_loss=damp_loss_dic[model_name][run.config.get('optim')][run.config.get('batch_size')][run.config.get('damping')]
                if test_loss < cur_loss:
                    damp_loss_dic[model_name][run.config.get('optim')][run.config.get('batch_size')][run.config.get('damping')]=test_loss

    print(damp_loss_dic)
    return damp_loss_dic

if __name__=='__main__':
    api = wandb.Api()
    optim_list = ['shampoo','kfac_mc','seng','kbfgs']
    batchsize_list = [32,512]
    damping_list = [1,1e-3,1e-6,1e-9,1e-12,1e-15]
    model_list=['mlp_512','resnet18']
    filename = './graph/damping_loss.png'

    sweep_list={
        'riverstone/grad_maker/u8thipfb',
        'riverstone/grad_maker/rm35hvs9',
    }

    damp_loss_dic=collect_runs(sweep_list)
    fig, axes = plt.subplots(nrows=1, ncols=len(model_list), figsize=(10, 5))

    for i in range(len(model_list)):
        model = model_list[i]
        for j in range(len(optim_list)):
            optim = optim_list[j]
            for bs in batchsize_list:
                dampli=list(damp_loss_dic[model][optim][bs].keys())
                lossli=list(damp_loss_dic[model][optim][bs].values())
                axes[i].plot(dampli,lossli,label=optim,color=col[optim],linestyle=lstyle[bs])
                axes[i].set_title(model)
                axes[i].set_xscale('log')
                axes[i].set_yscale('log')
                axes[i].set_xlabel('damping size')
                axes[i].set_ylabel('test_loss')
    
    plt.legend(loc='center left')
    plt.plot()
    plt.savefig(filename, dpi=300)
