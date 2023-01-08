import wandb
import matplotlib.pyplot as plt
import numpy as np

col = {'sgd':'tab:pink','psgd':'tab:red','kfac_mc':'tab:green','kfac_mc_local':'tab:cyan','seng':'tab:blue','shampoo':'tab:purple','adamw':'tab:brown','foof':'tab:olive','adahessian':'tab:gray'}
model_name_dic = {'mlp':'MLP','cnn':'CNN','resnet18':'Resnet18','vit_tiny_patch16_224':'ViT-tiny'}
optim_dict = {'sgd':'SGD','adamw':'AdamW','psgd':'PSGD(KF)','kfac_mc':'K-FAC(global)','kfac_mc_local':'K-FAC(local)','skfac_mc':'SK-FAC(1-mc)','shampoo':'Shampoo','seng':'SENG','smw_ngd':'SMW-NG','foof':'FOOF','adahessian':'AdaHessian'}

def make_dict(batchsize_list):
    acc_dic={}
    for optim in optim_list:
        acc_dic[optim]={}
        for bs in batchsize_list:
            acc_dic[optim][bs]=0
    return acc_dic

def remove_dict(acc_dic,batchsize_list):
    for optim in optim_list:
        for bs in batchsize_list:
            if error_rate:
                if acc_dic[optim][bs]>0.2:
                    acc_dic[optim].pop(bs)
            else:
                if acc_dic[optim][bs]<80:
                    acc_dic[optim].pop(bs)
    return acc_dic

def convert_to_error(acc_dic,batchsize_list):
    for optim in optim_list:
        for bs in batchsize_list:
            acc_dic[optim][bs]=(100-acc_dic[optim][bs])/100
    return acc_dic

def collect_runs(model,sweep_list_mlp,batchsize_list,metric='train_accuracy'):
    damp_loss_dic = make_dict(batchsize_list)
    for sweep_name in sweep_list_mlp:
        sweep = api.sweep(sweep_name)
        runs = sweep.runs
        for run in runs:
            model_name = run.config.get('model')
            if model_name == model  and run.config.get('batch_size') in batchsize_list and run.config.get('optim') in optim_list:
                if run.summary.get("cuda_max_memory") == -1 or type(run.summary.get(metric)) != float:
                    test_accuracy=0
                else:
                    test_accuracy=run.summary.get(metric)

                cur_loss=damp_loss_dic[run.config.get('optim')][run.config.get('batch_size')]
                if test_accuracy > cur_loss:
                    damp_loss_dic[run.config.get('optim')][run.config.get('batch_size')]=test_accuracy

                if run.config.get('optim')=='kfac_mc' and (run.config.get('ema_decay')==-1 or  run.config.get('ema_decay')==1):
                    cur_loss=damp_loss_dic['kfac_mc_local'][run.config.get('batch_size')]
                    if test_accuracy > cur_loss:
                        damp_loss_dic['kfac_mc_local'][run.config.get('batch_size')]=test_accuracy

    print(damp_loss_dic)
    return damp_loss_dic

if __name__=='__main__':
    api = wandb.Api()
    optim_list = ['sgd','adamw','shampoo','kfac_mc_local','kfac_mc','seng','psgd','foof','adahessian']
    batchsize_list = [256,512,1024,2048,4096,8192,16384]
    batchsize_list_res = [128,256,512,1024,2048,4096]
    batchsize_list_vit = [64,128,256,512,1024]
    with_clip=True
    without_clip=True
    error_rate=True
    ylabel_name='Test error rate'
    metric='test_accuracy'
    folder = './sotsuron/graph/'

    if error_rate:
        s=metric
        s.replace('accuracy','error')
        filename = folder+metric+'_bs.png'
    else:
        filename = folder+metric+'_bs.png'

    sweep_list_mlp={
        'riverstone/criteria/spzpahql',
        'riverstone/criteria/denix8xi',
        'riverstone/optcriteria/e11b9g2v',
        'riverstone/grad_maker/5sweoxrv',
        'riverstone/grad_maker/xb0y5kd5',
        'riverstone/grad_maker/n727u0s3',
        'riverstone/grad_maker/yqid66vz',
        'riverstone/grad_maker/8ufrctyz',
        'riverstone/grad_maker/p0s014j8',
        'riverstone/criteria/denix8xi',
        'riverstone/grad_maker/gqchvy20'
    }

    sweep_list_mlp_noclip={
        'riverstone/mnist_batch/3td3ceu8',
        'riverstone/mnist_batch/qw9psogu',
        'riverstone/mnist_batch/hmm2zrx6',
        'riverstone/mnist_batch/3y4chpxm'
    }

    sweep_list_vit={
        'riverstone/vit_batch/9q8eq6ql',
        'riverstone/vit_batch/zarall9n',
        'riverstone/vit_batch/9w0fl9qv',
        'riverstone/vit_batch/7qy5v4jk',
        'riverstone/vit_batch/tvbsauhx',
        'riverstone/vit_batch/qsdfynar',
        'riverstone/vit_batch/cpzqkh7f',
        'riverstone/vit_batch/npgaogw4',
        'riverstone/vit_batch/ltjonitr'
    }

    sweep_list_res={
        'riverstone/res_batch/lle7ao87',
        'riverstone/res_batch/swtoakmv',
        'riverstone/res_batch/fl4iqmyk',
        'riverstone/res_batch/fnetnblk',
        'riverstone/res_batch/2mro7urd',
        'riverstone/res_batch/xzh4rrxh',
        'riverstone/res_batch/zlh21342',
        'riverstone/res_batch/18fm6e4e'
    }

    if with_clip:
        if without_clip:
            sweep_list_mlp.union(sweep_list_mlp_noclip)
    else:
        if without_clip:
            sweep_list_mlp=sweep_list_mlp_noclip

    
    clip_test_dic_mlp=collect_runs('mlp',sweep_list_mlp,batchsize_list,metric=metric)
    clip_test_dic_res=collect_runs('resnet18',sweep_list_res,batchsize_list_res,metric=metric)
    clip_test_dic_vit=collect_runs('vit_tiny_patch16_224',sweep_list_vit,batchsize_list_vit,metric=metric)

    if error_rate:
        clip_test_dic_mlp=convert_to_error(clip_test_dic_mlp,batchsize_list)
        clip_test_dic_res=convert_to_error(clip_test_dic_res,batchsize_list_res)
        clip_test_dic_vit=convert_to_error(clip_test_dic_vit,batchsize_list_vit)

    clip_test_dic_res=remove_dict(clip_test_dic_res,batchsize_list_res)
    clip_test_dic_vit=remove_dict(clip_test_dic_vit,batchsize_list_vit)

    plt.rcParams["font.size"] = 28
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))

    for j in range(len(optim_list)):
        optim = optim_list[j]
        clip_dampli=list(clip_test_dic_vit[optim].keys())
        clip_lossli=list(clip_test_dic_vit[optim].values())
        axes[2].plot(clip_dampli,clip_lossli,label=optim_dict[optim],color=col[optim],marker='o')

        clip_dampli=list(clip_test_dic_res[optim].keys())
        clip_lossli=list(clip_test_dic_res[optim].values())
        axes[1].plot(clip_dampli,clip_lossli,label=optim_dict[optim],color=col[optim],marker='o')

        clip_dampli=list(clip_test_dic_mlp[optim].keys())
        clip_lossli=list(clip_test_dic_mlp[optim].values())
        axes[0].plot(clip_dampli,clip_lossli,label=optim_dict[optim],color=col[optim],marker='o')

        axes[0].set_xscale('log',base=2)
        axes[1].set_xscale('log',base=2)
        axes[2].set_xscale('log',base=2)
        
        if error_rate:
            axes[0].set_yscale('log',base=2)
            axes[1].set_yscale('log',base=2)
            axes[2].set_yscale('log',base=2)

        axes[0].set_xlabel('Batch size')
        axes[1].set_xlabel('Batch size')
        axes[2].set_xlabel('Batch size')
        axes[0].set_ylabel(ylabel_name)
        axes[1].set_ylabel(ylabel_name)
        axes[2].set_ylabel(ylabel_name)
        axes[0].set_title('MLP')
        axes[1].set_title('Resnet18')
        axes[2].set_title('ViT-T')
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.4, -0.1), ncol=5)
    plt.plot()
    plt.savefig(filename,dpi=80,bbox_inches='tight',pad_inches=0.1)
