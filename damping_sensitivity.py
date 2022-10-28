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

    sweep_clipping_list={
        'riverstone/grad_maker/u8thipfb',
        'riverstone/grad_maker/rm35hvs9',
        #'riverstone/grad_maker/9n9x42i8',
        #'riverstone/grad_maker/an9vomat'
    }

    sweep_noclipping_list={
        'riverstone/grad_maker/qi8038th',
        'riverstone/grad_maker/qp4porzm',
        #'riverstone/grad_maker/wd7asu5o',
        #'riverstone/grad_maker/83nitl9f'
    }

    clip_loss_dic=collect_runs(sweep_clipping_list)
    noclip_loss_dic=collect_runs(sweep_noclipping_list)
    fig, axes = plt.subplots(nrows=2, ncols=len(model_list), figsize=(10, 5))

    for i in range(len(model_list)):
        model = model_list[i]
        for j in range(len(optim_list)):
            optim = optim_list[j]
            for bs in batchsize_list:
                clip_dampli=list(clip_loss_dic[model][optim][bs].keys())
                clip_lossli=list(clip_loss_dic[model][optim][bs].values())

                noclip_dampli=list(noclip_loss_dic[model][optim][bs].keys())
                noclip_lossli=list(noclip_loss_dic[model][optim][bs].values())

                axes[0,i].plot(clip_dampli,clip_lossli,label=optim,color=col[optim],linestyle=lstyle[bs],marker='o')
                axes[1,i].plot(noclip_dampli,noclip_lossli,label=optim,color=col[optim],linestyle=lstyle[bs],marker='o')

                axes[0,i].set_title(model)
                axes[0,i].set_xscale('log')
                axes[0,i].set_yscale('log')
                axes[1,i].set_xscale('log')
                axes[1,i].set_yscale('log')

                axes[0,i].set_xlabel('damping')
                axes[1,i].set_xlabel('damping')
                axes[0,i].set_ylabel('test_loss with clipp')
                axes[1,i].set_ylabel('test_loss without clipp')
    
    plt.legend(loc='center left')
    plt.plot()
    plt.savefig(filename, dpi=300)
