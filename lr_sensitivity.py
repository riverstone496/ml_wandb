from matplotlib.lines import lineStyles
import wandb
import matplotlib.pyplot as plt

col = {'psgd':'r','kbfgs':'brown','kfac_mc':'green','seng':'cyan','shampoo':'purple'}
lstyle = {32:'dashed',512:'solid'}
api = wandb.Api()

# project内にrunが2万以上あり，検索に時間がかかるためsweep単位で指定．
def sweep_runs(model,optim,bs):
    if model == 'mlp'and optim in ['psgd','kfac_mc','seng','shampoo']:
        id = 'ytafdr5i'
    if model == 'mlp'and optim in ['kbfgs']:
        id = '5vg8z9fw'
    if model=='resnet18' and optim in ['psgd','kfac_mc','seng','shampoo']:
        id = 'lo2mhkf7'
    if model=='resnet18' and optim in ['kbfgs']:
        id = 'q41g33ln'
    if model=='vit_tiny' and optim in ['psgd','kfac_mc','seng','shampoo'] and bs == 32:
        id = '5snc12x1'
    if model=='vit_tiny' and optim in ['psgd','kfac_mc','seng','shampoo'] and bs == 512:
        id = 'ceu228qu'
    if model=='vit_tiny' and optim in ['kbfgs']:
        id = '4zn9b6v1'
    sweep_name = 'riverstone/grad_maker/' + id
    sweep = api.sweep(sweep_name)
    runs = sweep.runs
    return runs

def lr_acc_dict(model, optim, bs):
    lr_acc_dict = {}
    runs = sweep_runs(model,optim,bs)
    for run in runs:
        if run.config.get('optim') == optim and run.config.get('batch_size') == bs and run.config.get('epochs') == 20:
            if model != 'mlp':
                lr_acc_dict[run.config.get('lr')] = run.summary.get('test_loss')
            elif model == 'mlp' and run.config.get('width') == mlp_width:
                lr_acc_dict[run.config.get('lr')] = run.summary.get('test_loss')
    return dict(sorted(lr_acc_dict.items()))

if __name__=='__main__':
    filename='./graph/lr_loss.png'
    model_list = ['mlp','resnet18','vit_tiny']
    batchsize_list = [32,512]
    optim_list = ['shampoo','kfac_mc','psgd','seng','kbfgs']
    #optim_list=['kfac_mc']
    mlp_width = 512
    
    fig = plt.figure(figsize=(10,3))
    ax_list = []
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1)
        ax_list.append(ax)

    for i in range(len(model_list)):
        model = model_list[i]
        for optim in optim_list:
            for bs in batchsize_list:
                lr_acc = lr_acc_dict(model, optim, bs)
                lr_list = lr_acc.keys()
                acc_list = lr_acc.values()
                ax_list[i].plot(lr_list,acc_list,label=optim+' '+str(bs),color=col[optim],linestyle=lstyle[bs])
                
        ax_list[i].set_title(model)
        ax_list[i].set_xscale('log')
        ax_list[i].set_yscale('log')

    plt.legend()
    plt.plot()
    plt.savefig(filename)




