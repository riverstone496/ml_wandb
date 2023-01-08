import wandb
import matplotlib.pyplot as plt

api = wandb.Api()

def fetch_history(run,metric='test_accuracy',x='epoch'):
    out_dict={}
    if run.state == "finished":
        for i, row in run.history(keys=[metric,x]).iterrows():
            out_dict[int(row[x])]=row[metric]
    return out_dict    

runs=[
    'riverstone/criterion/jk610v0v',
    'riverstone/criterion/wp9p2utg',
    'riverstone/criterion/w0ge5qcx',
    'riverstone/criterion/fps0mzab',
    'riverstone/criterion/beplynpm',
]

plt.rcParams["font.size"] = 20
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
for run in runs:
    run = api.run(run)
    bs = run.config.get('batch_size')
    train_his_epoch = fetch_history(run,metric='hessian/one_sample.criterion3.all',x='epoch')
    train_his_iteration = fetch_history(run,metric='hessian/one_sample.criterion3.all',x='iteration')

    axes[0].plot(train_his_epoch.keys(), train_his_epoch.values(), label='Batch Size='+str(bs))
    axes[1].plot(train_his_iteration.keys(), train_his_iteration.values(), label='Batch Size='+str(bs))
    axes[0].set_xlabel('epoch')
    axes[1].set_xlabel('iteration')
    axes[0].set_ylabel('PSGD Criterion')
    axes[1].set_ylabel('PSGD Criterion')

plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.1), ncol=3)
plt.plot()
plt.savefig('sotsuron/graph/psgd_criterion.png',dpi=80,bbox_inches='tight',pad_inches=0.1)