import wandb
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

col = {'sgd':'tab:pink','psgd':'tab:red','kbfgs':'tab:brown','kfac_mc':'tab:green','seng':'tab:blue','shampoo':'tab:purple'}

def get_batchsize_list(model):
    if model == 'vit_tiny_imagenet' or model == 'mixer_imagenet':
        batchsize_list = [32,128,512]
    else:
        batchsize_list = [32,128,512,2048]
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
    def nonlinear_fit(x,a,b):
        return  a + b/x

    intlist=[]
    timelist=[]
    for interval in interval_list:
        if bs in thdic[interval]:
            intlist.append(interval)
            timelist.append(1/thdic[interval][bs])

    if len(intlist) <=1:
        return [],[]

    xlist = np.linspace(1, 300, 50)

    if optim=='sgd':
        t=min(timelist)
        time_list = np.array([t] * len(xlist))
    else:
        param, cov = curve_fit(nonlinear_fit, intlist, timelist)
        time_list = param[0]+param[1]/xlist

    thlist = 1/time_list
    return xlist,thlist

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
            if run.config.get('model') == 'mlp':
                model_name = 'mlp_'+str(run.config.get('width'))
            else:
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

def sgd_ratio(throughput_dic):
    throughput_ratio_dic = make_dict()
    for model in model_list:
        batchsize_list=get_batchsize_list(model)
        for optim in optim_list:
            for bs in batchsize_list:
                if throughput_dic[model]['sgd'][bs]!=0:
                    throughput_ratio_dic[model][optim][bs] = throughput_dic[model][optim][bs]/throughput_dic[model]['sgd'][bs]
                else:
                    throughput_ratio_dic[model][optim][bs] = 0

    return throughput_ratio_dic

if __name__=='__main__':
    api = wandb.Api()
    fixinterval=1
    fixbatchsize=512
    optim_list = ['sgd','shampoo','kfac_mc','psgd','seng','kbfgs']
    batch_size_list = [32,128,512,2048]
    interval_list=[1,3,10,30]
    model_list=['mlp_512','mlp_2048','resnet18','wideresnet28','vit_tiny_patch16_224','mixer_b16_224']
    filename = './graph/thmem.png'
    sweep_list={
        'riverstone/optprofiler/mz4j8kqs',
        'riverstone/optprofiler/v03bz3o4',
        'riverstone/optprofiler/lj9re5ab',
        'riverstone/optprofiler/xrflpbon',
        'riverstone/optprofiler/n71od3fe',
        'riverstone/optprofiler/51dt8cew',
        'riverstone/optprofiler/kjwxeo2i'
    }

    throughput_dic,memory_dic=collect_runs(sweep_list)
    #throughput_ratio_list = sgd_ratio(throughput_dic)
    throughput_dic=remove_zero(throughput_dic)
    memory_dic=remove_zero(memory_dic)

    fig, axes = plt.subplots(nrows=3, ncols=len(model_list), figsize=(32, 10))

    for i in range(len(model_list)):
        model = model_list[i]
        for j in range(len(optim_list)):
            optim = optim_list[j]

            bsli =list(throughput_dic[model][optim][fixinterval].keys())
            bsthli =list(throughput_dic[model][optim][fixinterval].values())

            intlist,inthlist=interval_time_list(throughput_dic[model][optim],fixbatchsize,optim)

            memli=list(memory_dic[model][optim][fixinterval].values())
            axes[0,i].plot(bsli,bsthli,label=optim,marker='o',color=col[optim])
            axes[1,i].plot(intlist,inthlist,label=optim,color=col[optim])
            axes[2,i].plot(bsli,memli,label=optim,marker='o',color=col[optim])

            axes[0,i].set_title(model)
            axes[0,i].set_xscale('log')
            axes[2,i].set_xscale('log')

            axes[0,i].set_xlabel('batch size')
            axes[1,i].set_xlabel('interval')
            axes[2,i].set_xlabel('batch size')
    axes[0,0].set_ylabel('throughput [image/s]')
    axes[1,0].set_ylabel('throughput [image/s]')
    axes[2,0].set_ylabel('memory [GB]')
    
    plt.legend(loc='center left', bbox_to_anchor=(1., .5))
    plt.plot()
    plt.savefig(filename, dpi=300)
