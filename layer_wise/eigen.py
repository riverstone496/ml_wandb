import wandb
import matplotlib.pyplot as plt

api = wandb.Api()

def collect_runs(run,metric='train_accuracy'):
    output_dic={}
    history = run.scan_history()
    history = [row for row in history]
    for metric_name in fetch_col(history):
        if metric not in metric_name:
            continue
        if 'all' in metric_name:
            continue
        train_his = fetch_history(history,metric_name)
        metric_name=metric_name.replace(metric, '')
        output_dic[metric_name]=train_his
    return output_dic

def plot(ax,his1,his2,his3,his4):
    for metric_name,train_his1 in his1.items():
        train_his2=his2[metric_name]
        train_his3=his3[metric_name]
        train_his4=his4[metric_name]
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

        ax[0].plot(train_his1.keys(), train_his1.values(),linestyle=linestyle,color=color,label=metric_name)
        ax[1].plot(train_his2.keys(), train_his2.values(),linestyle=linestyle,color=color,label=metric_name)
        ax[2].plot(train_his3.keys(), train_his3.values(),linestyle=linestyle,color=color,label=metric_name)
        ax[3].plot(train_his4.keys(), train_his4.values(),linestyle=linestyle,color=color,label=metric_name)

def fetch_col(history):
    key_set=set()
    for row in history:
        a=set(row.keys())
        key_set=key_set.union(a)
    key_set=list(key_set)
    key_set = sorted(key_set)
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

optim='kfac_mc'
epochs=50
damping=1
curvature_update_interval=10
lr=0.3

run='riverstone/criterion/wq4ttj5d'
run='riverstone/criterion/0mfd71bv'
run = api.run(run)
plt.rcParams["font.size"] = 26
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(40, 7))

dic1=collect_runs(run=run,metric='fisher/.criterion0.')
dic2=collect_runs(run=run,metric='fisher/.SGD_ratio_criterion2.')
dic3=collect_runs(run=run,metric='fisher/.SGD_ratio_criterion3.')
dic4=collect_runs(run=run,metric='matrix_info/P_inv_max_eigen_')
plot(axes,dic1,dic2,dic3,dic4)

axes[0].set_xlabel('Epoch')
axes[1].set_xlabel('Epoch')
axes[2].set_xlabel('Epoch')
axes[3].set_xlabel('Epoch')
axes[0].set_ylabel('Cos(x,Px)')
axes[1].set_ylabel('Criterion2 ratio with SGD')
axes[2].set_ylabel('Criterion3 ratio with SGD')
axes[3].set_ylabel('Max Eigen value of P Inverse')
#axes[0].set_yscale('log',base=2)
axes[1].set_yscale('log',base=2)
#axes[2].set_yscale('log',base=2)
axes[3].set_yscale('log',base=2)

plt.legend(loc='upper right', bbox_to_anchor=(0.8, -0.2), ncol=6)
plt.plot()
plt.savefig('layer_wise/graph/eigen.png',dpi=80,bbox_inches='tight',pad_inches=0.1)