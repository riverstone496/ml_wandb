import wandb
import matplotlib.pyplot as plt
import seaborn as sns

col = {'psgd':'r','kbfgs':'brown','kfac_mc':'green','seng':'cyan','shampoo':'purple'}
lstyle = {32:'dashed',512:'solid'}
api = wandb.Api()

mlp_sweep_id_list=[
    'f1if50ii',
    'k82xdkdk',
    'd5zvm08h',
    'ib9hp9ad',
    'daz5qb9t'
]

resnet_sweep_id_list=[
    '0yt3f6fd',
    '1lxkptqd',
    '0ml9c9n5',
    'nf77a1gi',
    '80vdlfza',
    'vtv9jhl2',
    'd3jgb9ls',
    'ao1zgb7m',
    'kvyztw4q',
    'sxtdbnwk'
]

vit_sweep_id_list=[
    'qrql589u',
    'u001u9on',
    's72klmfo',
    'im207a5p'
]

def make_dict(batchsize_list,interval_list):
    acc_dic={}
    for model in  model_list:
        acc_dic[model] = {}
        for optim in optim_list:
            acc_dic[model][optim]={}
            for bs in batchsize_list:
                acc_dic[model][optim][bs]={}
                for interval in interval_list:
                    acc_dic[model][optim][bs][interval]=0
    
    return acc_dic

def collect_runs(model):
    if model == 'resnet18':
        interval_list=[2,10,100]
        batchsize_list = [32,128,512,2048]
        id_list=resnet_sweep_id_list
    if model == 'mlp':
        interval_list=[1,10,100]
        batchsize_list = [32,128,512,2048]
        id_list=mlp_sweep_id_list
    if model == 'vit_tiny_imagenet':
        interval_list=[10,100]
        batchsize_list = [32,128,512]
        id_list=vit_sweep_id_list
    acc_dic = make_dict(batchsize_list,interval_list)

    for sweep_id in id_list:
        sweep_name = 'riverstone/grad_maker/' + sweep_id
        sweep = api.sweep(sweep_name)
        runs = sweep.runs

        for run in runs:
            if run.config.get('model') == 'mlp' and run.config.get('width') != mlp_width:
                continue
            if run.config.get('model') == model and run.config.get('epochs') == epoch and run.config.get('interval') in interval_list and run.config.get('batch_size') in batchsize_list and run.config.get('optim') in optim_list:
                if run.summary.get("cuda_max_memory") == -1 or type(run.summary.get("test_accuracy")) != float:
                    test_acc=0
                else:
                    test_acc=run.summary.get("test_accuracy")

                cur_acc = acc_dic[run.config.get('model')][run.config.get('optim')][run.config.get('batch_size')][run.config.get('interval')]
                if test_acc > cur_acc:
                    acc_dic[run.config.get('model')][run.config.get('optim')][run.config.get('batch_size')][run.config.get('interval')]=test_acc

    return acc_dic

if __name__=='__main__':
    filename='./graph/lr_loss.png'
    model_list = ['mlp','resnet18','vit_tiny_imagenet']
    optim_list = ['shampoo','kfac_mc','psgd','seng','kbfgs']
    mlp_width = 512
    epoch = 20

    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(25, 18))

    for i in range(len(model_list)):
        model = model_list[i]
        acc_dict = collect_runs(model)
        for j in range(len(optim_list)):
            optim = optim_list[j]
            batchsize_list = list(acc_dict[model][optim].keys())
            list_2d=[]
            for bs in batchsize_list:
                list_2d.append(list(acc_dict[model][optim][bs].values()))
            print(model,optim)
            print(list_2d)
            ilist = list(acc_dict[model][optim][bs].keys())
            sns.heatmap(list_2d,ax=axes[i, j], annot=True, cmap='hot',xticklabels=ilist,yticklabels=batchsize_list,fmt= '.4g')
            axes[i, j].set_title(optim)
            axes[i, j].set_xlabel('interval')
            axes[i, j].set_ylabel('batch size')

    plt.savefig('graph/seaborn_heatmap_list.png', dpi=600)
    plt.close('all')
