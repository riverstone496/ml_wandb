import numpy as np
import wandb
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

api = wandb.Api()
model_type = 'mixer'
model = 'mixer_s16_224'
#model = 'vit_tiny_patch16_224'
batch_size=32
filename = './graph/interval_'+model+'.png'

sweep_dic={
    'resnet':'riverstone/optprofiler/hof6qhuy',
    'wideresnet':'riverstone/optprofiler/88o1ai7y',
    'vit':'riverstone/optprofiler/y2zysvr9',
    'mixer':'riverstone/optprofiler/nfwp192i'
}
sweep_name = sweep_dic[model_type]


optim_list = ['shampoo','kfac_mc','psgd','kbfgs']
col = {'psgd':'tab:red','kbfgs':'tab:brown','kfac_mc':'tab:green','seng':'tab:cyan','shampoo':'tab:purple'}

sweep = api.sweep(sweep_name)
runs = sweep.runs

time_batch = {}
for opt in optim_list:
    time_batch[opt]={'interval':[],'time':[]}

for run in runs:
    if  run.config.get('model') == model and run.config.get('batch_size') == batch_size and run.config.get('optim') in optim_list:
        if run.summary.get("cuda_max_memory") == -1 or run.state!='finished' or type(run.summary.get("avg_step_time")) != float:
            time=0
        else:
            time=run.summary.get("avg_step_time")
            
            time_batch[run.config.get('optim')]['interval'].append(run.config.get("interval"))
            time_batch[run.config.get('optim')]['time'].append(time)

def nonlinear_fit(x,a,b):
    return  a + b/x

interval_list = np.linspace(1, 400, 50)
for opt in optim_list:
    array_x=time_batch[opt]['interval']
    array_y=time_batch[opt]['time']
    print(opt)
    param, cov = curve_fit(nonlinear_fit, array_x, array_y)
    time_list = param[0]+param[1]/interval_list
    plt.yscale('log')
    plt.plot(interval_list, time_list,label=opt,color=col[opt])

plt.legend(loc='upper right')
plt.title(model+'_'+str(batch_size))
plt.savefig(filename)

print(time_batch)