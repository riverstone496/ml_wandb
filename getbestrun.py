import wandb
api = wandb.Api()

sweep = api.sweep("riverstone/auto_asdl/6qg7cftm")
runs = sorted(sweep.runs,
key=lambda run: run.summary.get("train_loss", 0), reverse=True)
val_acc = runs[0].summary.get("train_loss", 0)
print(f"Best run {runs[0].name} with {val_acc}% validation accuracy")

print(runs[0])