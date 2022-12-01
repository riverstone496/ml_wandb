import wandb
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

col = {'sgd':'tab:pink','psgd':'tab:red','kfac_mc':'tab:green','seng':'tab:blue','shampoo':'tab:purple','skfac_mc':'tab:brown','smw_ngd':'tab:cyan'}
model_name_dic = {'mlp':'MLP','cnn':'CNN','resnet18':'Resnet18','vit_tiny_patch16_224':'ViT-tiny'}
optim_dict = {'sgd':'SGD','psgd':'PSGD(KF)','kfac_mc':'K-FAC(1-mc)','skfac_mc':'SK-FAC(1-mc)','shampoo':'Shampoo','seng':'SENG','smw_ngd':'SMW-NG'}

def get_batchsize_list(model):
    if model == 'vit_tiny_patch16_224':
        batchsize_list = [32,128,512]
    elif model == 'resnet18':
        batchsize_list = [32,64,128,256,512,1024,2048]
    else:
        batchsize_list = [32,64,128,256,512,1024,2048,4096,8192,16384,32768]
    return batchsize_list

def make_dict():
    acc_dic={}
    for model in  model_list:
        acc_dic[model] = {}
        batchsize_list=get_batchsize_list(model)
        for optim in optim_list:
            acc_dic[model][optim]={}
            for interval in interval_list:
                    acc_dic[model][optim][interval]={}
                    for bs in batchsize_list:
                        acc_dic[model][optim][interval][bs]=0
    return acc_dic

def interval_time_list(thdic,bs,optim):
    intlist=[]
    timelist=[]
    for interval in interval_list:
        if bs in thdic[optim][interval] and thdic['sgd'][interval][bs]!=0 and thdic[optim][interval][bs]!=0:
            intlist.append(interval)
            if optim != 'smw_ngd':
                timelist.append(thdic[optim][interval][bs]/thdic['sgd'][interval][bs])
            else:
                timelist.append(thdic[optim][1][bs]/thdic['sgd'][1][bs])
    return intlist,timelist

def remove_zero(acc_dic):
    for model in  model_list:
        batchsize_list=get_batchsize_list(model)
        for optim in optim_list:
            for interval in interval_list:
                for bs in batchsize_list:
                    if acc_dic[model][optim][interval][bs]==0:
                        acc_dic[model][optim][interval].pop(bs)
    return acc_dic

def collect_runs(sweep_list):
    throughput_dic = make_dict()
    memory_dic = make_dict()

    for sweep_name in sweep_list:
        sweep = api.sweep(sweep_name)
        runs = sweep.runs

        for run in runs:
            model_name = run.config.get('model')
            batchsize_list=get_batchsize_list(model_name)

            if model_name in model_list  and run.config.get('batch_size') in batchsize_list and run.config.get('optim') in optim_list:
                if run.summary.get("cuda_max_memory") == -1 or type(run.summary.get("avg_step_time")) != float:
                    throughput=0
                    mem=0
                else:
                    throughput=run.config.get('batch_size')/run.summary.get("avg_step_time")
                    mem=run.summary.get("cuda_max_memory")/(1024*1024*1024)

                throughput_dic[model_name][run.config.get('optim')][run.config.get('interval')][run.config.get('batch_size')]   =throughput
                memory_dic[model_name][run.config.get('optim')][run.config.get('interval')][run.config.get('batch_size')]       =mem

    return throughput_dic,memory_dic

def sgd_ratio(throughput_dic,cutone=False):
    throughput_ratio_dic = make_dict()
    for model in model_list:
        batchsize_list=get_batchsize_list(model)
        for optim in optim_list:
            for bs in batchsize_list:
                if throughput_dic[model]['sgd'][1][bs]!=0:
                    throughput_ratio_dic[model][optim][1][bs] = throughput_dic[model][optim][1][bs]/throughput_dic[model]['sgd'][1][bs]
                    if cutone and throughput_ratio_dic[model][optim][1][bs]>=1:
                        throughput_ratio_dic[model][optim][1][bs]=1
                else:
                    throughput_ratio_dic[model][optim][1][bs] = 0

    return throughput_ratio_dic

if __name__=='__main__':
    api = wandb.Api()
    fixinterval=1
    fixbatchsize=128
    optim_list = ['sgd','shampoo','kfac_mc','skfac_mc','psgd','seng','smw_ngd']
    batch_size_list = [32,128,512,2048]
    interval_list=[1,3,10,30,100]
    model_list=['mlp','cnn','resnet18','vit_tiny_patch16_224']
    filename = './graph/ipsj1.png'
    sweep_list={
        'riverstone/optprofiler/vtvb4lpb',
        'riverstone/optprofiler/ulw4fmpv',#後で下と入れ替える
        #https://wandb.ai/riverstone/optprofiler/sweeps/ehvm0mls?workspace=user-riverstone,
        'riverstone/optprofiler/ltxv1app',
        'riverstone/optprofiler/51dt8cew',
    }

    throughput_dic,memory_dic=collect_runs(sweep_list)

    throughput_ratio_list = sgd_ratio(throughput_dic,cutone=True)
    memory_ratio_dic = sgd_ratio(memory_dic)

    throughput_ratio_list=remove_zero(throughput_ratio_list)
    memory_ratio_dic=remove_zero(memory_ratio_dic)

    plt.rcParams["font.size"] = 20
    dpi=300
    fig, axes = plt.subplots(nrows=3, ncols=len(model_list), figsize=(28, 18))

    for i in range(len(model_list)):
        model = model_list[i]
        for j in range(len(optim_list)):
            optim = optim_list[j]

            bsli =list(throughput_ratio_list[model][optim][fixinterval].keys())
            bsthli =list(throughput_ratio_list[model][optim][fixinterval].values())

            intlist,inthlist=interval_time_list(throughput_dic[model],fixbatchsize,optim)

            memli=list(memory_ratio_dic[model][optim][fixinterval].values())
            axes[0,i].plot(bsli,bsthli,label=optim_dict[optim],marker='o',color=col[optim])
            axes[2,i].plot(intlist,inthlist,label=optim_dict[optim],marker='o',color=col[optim])
            axes[1,i].plot(bsli,memli,label=optim_dict[optim],marker='o',color=col[optim])

            axes[0,i].set_title(model_name_dic[model])
            axes[0,i].set_xscale('log',base=2)
            axes[1,i].set_xscale('log',base=2)
            axes[0,i].set_yscale('log',base=2)
            axes[1,i].set_yscale('log',base=2)

            axes[2,i].set_xscale('log',base=3)
            axes[2,i].set_yscale('log',base=3)

            axes[0,i].set_xlabel('Batch size')
            axes[2,i].set_xlabel('Interval')
            axes[1,i].set_xlabel('Batch size')
    axes[0,0].set_ylabel('Throughput vs SGD')
    axes[2,0].set_ylabel('Throughput vs SGD')
    axes[1,0].set_ylabel('Memory vs SGD')
    
    plt.legend(loc='upper center', bbox_to_anchor=(-1, -0.2), ncol=8)
    plt.plot()
    plt.savefig(filename,dpi=500,bbox_inches='tight',pad_inches=0.1)
