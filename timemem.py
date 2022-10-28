import wandb
import matplotlib.pyplot as plt
import numpy as np

col = {'psgd':'tab:red','kbfgs':'tab:brown','kfac_mc':'tab:green','seng':'tab:blue','shampoo':'tab:purple'}

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
            for bs in batchsize_list:
                acc_dic[model][optim][bs]=0
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

            #if run.config.get('interval') != interval and run.config.get('optim') != 'sgd':
            #    continue

            if run.config.get('interval') == interval and model_name in model_list  and run.config.get('batch_size') in batchsize_list and run.config.get('optim') in optim_list:
                if run.summary.get("cuda_max_memory") == -1 or type(run.summary.get("avg_step_time")) != float:
                    throughput=0
                    mem=0
                else:
                    throughput=run.config.get('batch_size')/run.summary.get("avg_step_time")
                    mem=run.summary.get("cuda_max_memory")/(1024*1024*1024)

                throughput_dic[model_name][run.config.get('optim')][run.config.get('batch_size')]=throughput
                memory_dic[model_name][run.config.get('optim')][run.config.get('batch_size')]=mem

    return throughput_dic,memory_dic

def sgd_ratio(throughput_dic):
    print(throughput_dic)
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
    interval=1
    optim_list = ['shampoo','kfac_mc','psgd','seng','kbfgs']
    batch_size_list = [32,128,512,2048]
    model_list=['mlp_128','mlp_512','mlp_2048','resnet18','wideresnet28','vit_tiny_patch16_224','mixer_b16_224']
    filename = './graph/thmem.png'
    sweep_list={
        'riverstone/optprofiler/mz4j8kqs',
        'riverstone/optprofiler/v03bz3o4',
        'riverstone/optprofiler/lj9re5ab',
        'riverstone/optprofiler/zmgg87bc',
        'riverstone/optprofiler/fs9juimy',
        'riverstone/optprofiler/51dt8cew',
        'riverstone/optprofiler/kjwxeo2i'
    }

    throughput_dic,memory_dic=collect_runs(sweep_list)
    #throughput_ratio_list = sgd_ratio(throughput_dic)

    fig, axes = plt.subplots(nrows=2, ncols=len(model_list), figsize=(30, 30))

    for i in range(len(model_list)):
        model = model_list[i]
        for j in range(len(optim_list)):
            optim = optim_list[j]

            bsli=list(throughput_dic[model][optim].keys())
            thli=list(throughput_dic[model][optim].values())
            memli=list(memory_dic[model][optim].values())
            axes[0,i].plot(bsli,thli,label=optim)
            axes[1,i].plot(bsli,memli,label=optim)

            axes[0,i].set_title(model)
            axes[0,i].set_xscale('log')
            axes[1,i].set_xscale('log')

            axes[0,i].set_xlabel('batch size')
            axes[1,i].set_xlabel('batch size')
            axes[0,i].set_ylabel('throughput')
            axes[1,i].set_ylabel('memory')
    
    plt.legend(loc='center left', bbox_to_anchor=(1., .5))
    plt.plot()
    plt.savefig(filename, dpi=300)
