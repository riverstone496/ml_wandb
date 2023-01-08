import wandb
import matplotlib.pyplot as plt

api = wandb.Api()

def fetch_history(run,metric='test_accuracy'):
    run = api.run(run)
    out_dict={}
    if run.state == "finished":
        for i, row in run.history(keys=["test_accuracy",'epoch']).iterrows():
            out_dict[row["epoch"]]=row["test_accuracy"]
    return out_dict

def plot_curve(X,Y):
    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("epochs")
    plt.ylabel("Test Acc")
    plt.plot(X, Y, color="tab:blue", label="Test Acc")
    plt.show()


run="riverstone/criterion/1s5pb65n"
train_his = fetch_history(run)
plot_curve(train_his.keys(),train_his.values())