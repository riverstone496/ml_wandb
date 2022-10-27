import wandb
import matplotlib.pyplot as plt
import numpy as np

api = wandb.Api()
interval=30
model = 'mixer'
optim_list = ['shampoo','kfac_mc','psgd','seng','kbfgs']
batch_size_list = [32,128,512,2048]
model_list = ['mixer_s16_224','mixer_b16_224','mixer_l16_224']
#model_list=['vit_tiny_patch16_224','vit_small_patch16_224','vit_base_patch16_224','vit_large_patch16_224']
#model_list=['wideresnet16','wideresnet28','wideresnet40']
#model_list=['resnet18','resnet34','resnet50','resnet101']

memory_filename = './graph/mixer_memory.png'
time_filename   = './graph/mixer_time.png'

sweep_dic={
    'resnet':'riverstone/optprofiler/hof6qhuy',
    'wideresnet':'riverstone/optprofiler/88o1ai7y',
    'vit':'riverstone/optprofiler/y2zysvr9',
    'mixer':'riverstone/optprofiler/nfwp192i'
}
sweep_name = sweep_dic[model]

col = {'psgd':'tab:red','kbfgs':'tab:brown','kfac_mc':'tab:green','seng':'tab:blue','shampoo':'tab:purple'}

sweep = api.sweep(sweep_name)
runs = sweep.runs

time_batch ,memory_batch= {},{}
for model in model_list:
    time_batch[model]={}
    memory_batch[model] = {}
    for opt in optim_list:
        time_batch[model][opt] = {}
        memory_batch[model][opt] = {}

for run in runs:
    if  run.config.get('interval') == interval and run.config.get('model') in model_list and run.config.get('batch_size') in batch_size_list:
        if run.summary.get("cuda_max_memory") == -1 or run.state!='finished' or type(run.summary.get("avg_step_time")) != float:
            time=0
            mem=0
        else:
            time=run.summary.get("avg_step_time")
            mem=run.summary.get("cuda_max_memory")
        time_batch[run.config.get('model')][run.config.get('optim')][run.config.get('batch_size')]=time
        memory_batch[run.config.get('model')][run.config.get('optim')][run.config.get('batch_size')]=mem

def make_graph(batch_list,filename):
    fig = plt.figure(figsize=(10,10))
    ax_list = []
    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1)
        ax_list.append(ax)
    width = 0.1
    left = np.arange(len(batch_size_list))

    for i in range(len(model_list)):
        model = model_list[i]
        count=0
        for opt,bmem in batch_list[model].items():
            bmem = dict(sorted(bmem.items()))
            mem = bmem.values()

            print(opt,mem)
            ax_list[i].bar(left+width*count, mem, width=width, align='center',label=opt,color=col[opt])
            count+=1

        ax_list[i].legend()
        ax_list[i].set_xticks([0.25,1.25,2.25,3.25]) 
        ax_list[i].set_xticklabels(batch_size_list)
        ax_list[i].set_title(model+' interval:'+str(interval))
    
    plt.yscale('log')
    plt.savefig(filename)

make_graph(memory_batch ,memory_filename)
make_graph(time_batch   ,time_filename)

