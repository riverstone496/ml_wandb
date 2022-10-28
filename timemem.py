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

            if model_name in model_list and run.config.get('interval') == interval and run.config.get('batch_size') in batchsize_list and run.config.get('optim') in optim_list:
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
    interval=3
    optim_list = ['sgd','shampoo','kfac_mc','psgd','seng','kbfgs']
    batch_size_list = [32,128,512,2048]
    model_list=['mlp_128','mlp_512','mlp_2048','resnet18','wideresnet28','vit_tiny_patch16_224','mixer_b16_224']

    filename = './graph/thmem.png'

    sweep_dic={
        'resnet':'riverstone/optprofiler/hof6qhuy',
        'wideresnet':'riverstone/optprofiler/88o1ai7y',
        'vit':'riverstone/optprofiler/y2zysvr9',
        'mixer':'riverstone/optprofiler/nfwp192i',
        'sgd_mlp':'riverstone/optprofiler/2qyu4jk0',
        'sgd':'riverstone/optprofiler/6enkjejo'
    }

    throughput_dic,memory_dic=collect_runs(list(sweep_dic.values()))
    throughput_ratio_list = sgd_ratio(throughput_dic)

    fig, axes = plt.subplots(nrows=2, ncols=len(model_list), figsize=(25, 18))

    for i in range(len(model_list)):
        model = model_list[i]
        for j in range(len(optim_list)):
            optim = optim_list[j]

            bsli=list(throughput_ratio_list[model][optim].keys())
            thli=list(throughput_ratio_list[model][optim].values())
            memli=list(memory_dic[model][optim].values())
            plt.plot(bsli,thli,ax=axes[i, 0])
            plt.plot(bsli,memli,ax=axes[i, 1])

        axes[i, 0].set_title(optim)
        axes[i, 0].set_xlabel('batch size')
        axes[i, 1].set_xlabel('batch size')
        axes[i, 0].set_ylabel('SGD throughput ratio')
        axes[i, 1].set_ylabel('memory')
    
    plt.legend(loc='center left', bbox_to_anchor=(1., .5))
    plt.plot()
    plt.savefig(filename, dpi=300)
