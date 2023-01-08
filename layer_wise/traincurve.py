import wandb
import matplotlib.pyplot as plt

api = wandb.Api()

def collect_runs(ax,sweep_name,metric='train_accuracy'):
    sweep = api.sweep(sweep_name)
    runs = sweep.runs
    for run in runs:
        if run.state != "finished":
            continue
        if run.config.get('optim') != optim or run.config.get('epochs') != epochs or run.config.get('damping') != damping or run.config.get('curvature_update_interval') != curvature_update_interval or run.config.get('lr') != lr:
            continue
        precond_module = run.config.get('precond_module_name')
        precond_module=precond_module.split(',')
        if len(precond_module) >=3:
            continue
        train_his = fetch_history(run,metric)

        if 'linear' in precond_module:
            linestyle='solid'
        else:
            linestyle='dashed'
        ax.plot(train_his.keys(), train_his.values(), linestyle=linestyle)

def fetch_history(run,metric='test_accuracy'):
    out_dict={}
    if run.state == "finished":
        for i, row in run.history(keys=[metric,'epoch']).iterrows():
            out_dict[row["epoch"]]=row[metric]
    return out_dict

optim='kfac_mc'
epochs=50
damping=1
curvature_update_interval=10
lr=0.3

sweep='riverstone/skip_modules/257qih76'
plt.rcParams["font.size"] = 20
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))

collect_runs(axes[0],sweep_name=sweep,metric='train_accuracy')
collect_runs(axes[1],sweep_name=sweep,metric='test_accuracy')
axes[0].set_xlabel('Epoch')
axes[1].set_xlabel('Epoch')
axes[0].set_ylabel('Train Acc')
axes[1].set_ylabel('Test Acc')

plt.plot()
plt.savefig('sotsuron/graph/layerwise.png',dpi=80,bbox_inches='tight',pad_inches=0.1)