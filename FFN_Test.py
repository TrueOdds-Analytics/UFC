import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
from optuna.trial import TrialState
import optuna.importance
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

best_accuracy = 0
best_model_state = None
best_train_accuracy = None
best_train_losses = None
best_test_accuracy = None
best_test_losses = None


class FFNnet(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, dropout):
        super().__init__()

        self.hidden_layers = nn.ModuleList()
        prev_size = input_size

        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(prev_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
            x = self.dropout(x)

        output = self.out(x)
        return output


def define_model(trial):
    input_size = 292
    output_size = 1
    num_epochs = 500

    batch_size = trial.suggest_categorical('batch_size', [32])
    dropout = trial.suggest_float('dropout', 0.3, 0.8)
    lr = trial.suggest_float("lr", 1e-06, 1e-04)

    num_layers = trial.suggest_int("num_layers", 1, 3)
    hidden_sizes = [trial.suggest_int(f"hidden_size_{i}", 32, 64) for i in range(num_layers)]

    net = FFNnet(input_size, output_size, hidden_sizes, dropout)
    return net, lr, batch_size, num_epochs


def objective(trial):
    global best_accuracy, best_model_state, best_train_accuracy, best_train_losses, best_test_accuracy, best_test_losses

    l_lambda = 1e-04
    net, lr, batch_size, num_epochs = define_model(trial)
    net = net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.1)
    loss_function = nn.BCEWithLogitsLoss()

    train_loader, test_loader = get_loaders(batch_size)

    trial_start = datetime.now()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, min_lr=1e-7)

    train_accuracy, train_losses, test_accuracy, test_losses = trainModel(
        num_epochs, net, optimizer, loss_function, train_loader, test_loader, device, l_lambda, scheduler, trial)

    print("\n")
    print(f"Test Loss: {test_losses.min()}")
    print(f"Test Accuracy: {test_accuracy.max()}")

    max_test_accuracy = np.max(test_accuracy)
    max_index = np.argmax(test_accuracy)
    plot_results(trial.number, num_epochs, train_accuracy, test_accuracy, train_losses, test_losses, max_index,
                 test_losses.min(), max_test_accuracy)

    if max_test_accuracy > best_accuracy or best_accuracy == 0:
        best_accuracy = max_test_accuracy
        best_model_state = net.state_dict().copy()
        best_train_accuracy = train_accuracy
        best_train_losses = train_losses
        best_test_accuracy = test_accuracy
        best_test_losses = test_losses

    if best_accuracy == max_test_accuracy:
        torch.save(best_model_state,
                   f"models/trial_{trial.number}_epoch_{max_index + 1}_test_accuracy_{max_test_accuracy:.4f}_test_loss_{test_losses.min():.4f}.pth")

    trial_duration = datetime.now() - trial_start
    if trial_duration.total_seconds() > 6000:
        raise optuna.exceptions.TrialPruned()

    return max_test_accuracy


def trainModel(num_epochs, net, optimizer, loss_function, train_data, test_data, device, l1_lambda, scheduler,
               trial=None):
    train_accuracy = np.zeros(num_epochs)
    train_losses = np.zeros(num_epochs)
    test_accuracy = np.zeros(num_epochs)
    test_losses = np.zeros(num_epochs)

    for epochi in range(num_epochs):
        net.train()

        segment_loss = []
        segment_accuracy = []
        for X, y in train_data:
            X = X.to(device)
            y = y.to(device)

            output = net(X)
            loss = loss_function(output, y)

            l1_penalty = sum(p.abs().sum() for p in net.parameters())

            total_loss = loss + l1_lambda * l1_penalty

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
            optimizer.step()

            predicted = (output > 0.0).float()
            acc = (predicted == y).float().mean() * 100

            segment_loss.append(total_loss.item())
            segment_accuracy.append(acc.item())

        train_losses[epochi] = np.mean(segment_loss)
        train_accuracy[epochi] = np.mean(segment_accuracy)

        test_loss = []
        test_acc = []

        with torch.no_grad():
            net.eval()
            for X, y in test_data:
                X = X.to(device)
                y = y.to(device)

                output = net(X)

                loss = loss_function(output, y)

                l1_penalty = sum(p.abs().sum() for p in net.parameters())

                total_loss = loss + l1_lambda * l1_penalty

                predicted = (output > 0.0).float()
                acc = (predicted == y).float().mean() * 100

                test_loss.append(total_loss.item())
                test_acc.append(acc.item())

            test_losses[epochi] = np.mean(test_loss)
            test_accuracy[epochi] = np.mean(test_acc)

        scheduler.step(np.mean(test_loss))

        if trial is not None:
            trial.report(test_accuracy[epochi], epochi)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        print(f"\rFinished epoch: {epochi + 1}/{num_epochs}", end="", flush=True)

    return train_accuracy, train_losses, test_accuracy, test_losses


def get_loaders(batch_size):
    # Load train data from CSV
    train_data = pd.read_csv('data/train_data.csv')
    train_labels = train_data['winner']
    train_data = train_data.drop(['winner'], axis=1)

    # Shuffle train data
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    train_labels = train_labels.sample(frac=1).reset_index(drop=True)

    # Load test data from CSV
    test_data = pd.read_csv('data/test_data.csv')
    test_labels = test_data['winner']
    test_data = test_data.drop(['winner'], axis=1)

    # Shuffle test data
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    test_labels = test_labels.sample(frac=1).reset_index(drop=True)

    test_dataT = torch.tensor(test_data.values).float()
    test_labelsT = torch.tensor(test_labels.values).float().unsqueeze(1)

    train_dataT = torch.tensor(train_data.values).float()
    train_labelsT = torch.tensor(train_labels.values).float().unsqueeze(1)

    train_dataset = TensorDataset(train_dataT, train_labelsT)
    test_dataset = TensorDataset(test_dataT, test_labelsT)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader


def plot_results(trial_number, num_epochs, train_accuracy, test_accuracy, train_losses, test_losses, max_index,
                 min_test_loss, max_test_accuracy):
    dpi = 300
    width_inch = 3840 / dpi
    height_inch = 2160 / dpi

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width_inch, height_inch), dpi=dpi)

    ax1.plot(range(1, num_epochs + 1), train_accuracy, label='Train Accuracy')
    ax1.plot(range(1, num_epochs + 1), test_accuracy, label='Test Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy')
    ax1.legend()

    ax2.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    ax2.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Losses')
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig.suptitle(
        f'Trial {trial_number} - Epoch {max_index + 1} - Test Loss: {min_test_loss:.4f}, Test Accuracy: {max_test_accuracy:.4f}',
        fontsize=16)

    plt.show()


def balance_classes(data, label_column):
    print("Length of classes before balancing:")
    print(data[label_column].value_counts())

    X = data.drop(label_column, axis=1)
    y = data[label_column]

    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)

    df_balanced = pd.concat([pd.DataFrame(X_res, columns=X.columns),
                             pd.DataFrame(y_res, columns=[label_column])], axis=1)

    new_data_points = len(df_balanced) - len(data)
    print(f"Number of new upsampled data points created by SMOTE: {new_data_points}")

    print("Length of classes after balancing:")
    print(df_balanced[label_column].value_counts())

    return df_balanced.drop(label_column, axis=1), df_balanced[label_column]


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    study = optuna.create_study(direction="maximize",
                                pruner=optuna.pruners.MedianPruner(),
                                sampler=optuna.samplers.TPESampler(),
                                study_name="stronger l1 reg")
    study.optimize(objective, n_trials=10)
    best_accuracy = 0
    best_model_state = None
    best_train_accuracy = None
    best_train_losses = None
    best_test_accuracy = None
    best_test_losses = None

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    param_importances = optuna.importance.get_param_importances(study)
    print("Parameter Importance:")
    for param, importance in param_importances.items():
        print(f"  {param}: {importance}")