import wandb
import matplotlib.pyplot as plt
import numpy as np

api = wandb.Api()
interval=30
model_list=['mixer_s16_224','mixer_b16_224','mixer_l16_224']
optim_list = ['shampoo','kfac_mc','psgd','seng','kbfgs']

sweep = api.sweep("riverstone/optprofiler/cncgp206")
runs = sweep.runs

time_batch ,memory_batch= {},{}
for model in model_list:
    time_batch[model]={}
    memory_batch[model] = {}
    for opt in optim_list:
        time_batch[model][opt] = {}
        memory_batch[model][opt] = {}

for run in runs:
    if  run.config.get('interval') == interval:
        time_batch[run.config.get('model')][run.config.get('optim')][run.config.get('batch_size')]=run.summary.get("avg_step_time")
        memory_batch[run.config.get('model')][run.config.get('optim')][run.config.get('batch_size')]=run.summary.get("cuda_max_memory")


batch_size = [32,128,512,2048]
fig = plt.figure(figsize=(10,10))
ax_list = []
for i in range(4):
    ax = fig.add_subplot(2, 2, i+1)
    ax_list.append(ax)
width = 0.1
left = np.arange(len(batch_size))
count=0
for i in range(len(model_list)):
    model = model_list[i]
    for opt,bmem in memory_batch[model].items():
        bmem = dict(sorted(bmem.items()))
        mem = bmem.values()
        ax_list[i].bar(left+width*count, mem, width=width, align='center',label=opt)
        count+=1
    ax_list[i].legend()
    ax_list[i].set_title(model)

plt.savefig('./graph/'+model+'_memory.png')