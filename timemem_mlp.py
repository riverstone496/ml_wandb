import wandb
import matplotlib.pyplot as plt
import numpy as np

col = {'psgd':'r','kbfgs':'brown','kfac_mc':'green','seng':'cyan','shampoo':'purple'}

api = wandb.Api()
interval=3
sweep_name = "riverstone/optprofiler/lr3y686k"
width_list=[128,512,2048]
optim_list = ['shampoo','kfac_mc','psgd','seng','kbfgs']

sweep = api.sweep(sweep_name)
runs = sweep.runs

time_batch ,memory_batch= {},{}
for width in width_list:
    time_batch[width]={}
    memory_batch[width] = {}
    for opt in optim_list:
        time_batch[width][opt] = {}
        memory_batch[width][opt] = {}

for run in runs:
    if  run.config.get('interval') == interval and run.config.get('width') in width_list:
        if run.summary.get("cuda_max_memory") == -1:
            time=0
            mem=0
        else:
            time=run.summary.get("avg_step_time")
            mem=run.summary.get("cuda_max_memory")
        time_batch[run.config.get('width')][run.config.get('optim')][run.config.get('batch_size')]=time
        memory_batch[run.config.get('width')][run.config.get('optim')][run.config.get('batch_size')]=mem

def make_graph(batch_list,filename):
    batch_size = [32,128,512,2048]
    fig = plt.figure(figsize=(10,10))
    ax_list = []
    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1)
        ax_list.append(ax)
    width = 0.1
    left = np.arange(len(batch_size))

    for i in range(len(width_list)):
        width = width_list[i]
        count=0
        for opt,bmem in batch_list[width].items():
            bmem = dict(sorted(bmem.items()))
            mem = bmem.values()

            ax_list[i].bar(left+width*count, mem, width=width, align='center',label=opt)
            count+=1
            
        ax_list[i].legend()
        ax_list[i].set_xticks([0.25,1.25,2.25,3.25]) 
        ax_list[i].set_xticklabels(batch_size)
        ax_list[i].set_title(width)
    
    plt.savefig(filename)

make_graph(memory_batch ,'./graph/mlp_memory.png')
make_graph(time_batch   ,'./graph/mlp_time.png')

