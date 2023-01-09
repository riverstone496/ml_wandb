import wandb
import matplotlib.pyplot as plt

api = wandb.Api()

def collect_runs(run,metric='train_accuracy',def_metic=False):
    output_dic={}
    history = run.scan_history()
    history = [row for row in history]
    for metric_name in fetch_col(history):
        if 'all' in metric_name:
            continue
        if metric not in metric_name:   
            continue

        train_his = fetch_history(history,metric_name)
        if def_metic:
            def_metric_name = metric_name.replace('sensitivity/','sensitivity_sgd/')
            train_his_def = fetch_history(history,def_metric_name)
            train_his = minus_dic(train_his,train_his_def)

        metric_name=metric_name.replace(metric, '')
        output_dic[metric_name]=train_his
    return output_dic

def plot(ax,his_list):
    for metric_name in his_list[0].keys():
        for i in range(len(his_list)):
            train_his = his_list[i][metric_name]
            if 'conv1' in metric_name:
                linestyle='solid'
            elif 'conv2' in metric_name:
                linestyle='dashed'
            elif 'shortcut' in metric_name:
                linestyle='dotted'
            else:
                linestyle='solid'

            if 'layer1' in metric_name or 'block1' in metric_name:
                color = 'tab:cyan'
            elif 'layer2' in metric_name or 'block2' in metric_name:
                color = 'tab:green'
            elif 'layer3' in metric_name or 'block3' in metric_name:
                color = 'tab:blue'
            elif 'layer4' in metric_name:
                color = 'tab:purple'
            elif 'linear' in metric_name or 'fc' in metric_name:
                linestyle='solid'
                color = 'tab:orange' 
            else:
                color = 'tab:red'
            ax[i].plot(train_his.keys(), train_his.values(),linestyle=linestyle,color=color,label=metric_name)

def fetch_col(history):
    key_set=set()
    for row in history:
        a=set(row.keys())
        key_set=key_set.union(a)
    key_set=list(key_set)
    key_set = sorted(key_set,)
    return key_set

def fetch_history(history,metric='test_accuracy'):
    out_dict={}
    for row in history:
        if metric not in row.keys():
            continue
        if row["epoch"]>epochs or row["epoch"]==1:
            continue
        out_dict[row["epoch"]]=row[metric]
    return out_dict

def minus_dic(dic1,dic2):
    out_dic={}
    for (k,v) in dic1.items():
        if dic1[k]-dic2[k]>=0:
            out_dic[k] = dic1[k]-dic2[k]
    return out_dic

optim='shampoo'
epochs=50
damping=1
curvature_update_interval=10
lr=0.3

run='riverstone/sensitivity/f07u9936'
#run='riverstone/sensitivity/np4nqrn1' #psgd
run='riverstone/sensitivity/vbu91x3i' #shampoo

run = api.run(run)
plt.rcParams["font.size"] = 20
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 6))

dic1=collect_runs(run=run,metric='hessian_sensitivity/.')
dic2=collect_runs(run=run,metric='hessian_sensitivity/.',def_metic=True)
#dic3=collect_runs(run=run,metric='simple_sensitivity/.')
#dic4=collect_runs(run=run,metric='kl_sensitivity/.')
plot(axes,[dic1,dic2,dic2])

axes[0].set_xlabel('Epoch')
axes[1].set_xlabel('Epoch')
axes[2].set_xlabel('Epoch')
#axes[3].set_xlabel('Epoch')
axes[0].set_ylabel('Hessian Sensitivity')
axes[1].set_ylabel('Hessian Sensitivity dif with SGD')
axes[2].set_ylabel('Hessian Sensitivity dif with SGD')

#axes[2].set_ylabel('Simple sensitivity')
#axes[3].set_ylabel('KL sensitivity')
axes[0].set_yscale('log',base=2)
#axes[1].set_yscale('log',base=2)
axes[2].set_yscale('log',base=2)
#axes[3].set_yscale('log',base=2)

plt.legend(loc='upper right', bbox_to_anchor=(1, -0.3), ncol=6)
plt.plot()
plt.savefig('layer_wise/graph/sensitivity_res_'+optim+'.png',dpi=80,bbox_inches='tight',pad_inches=0.1)