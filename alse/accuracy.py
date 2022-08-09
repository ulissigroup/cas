import torch


def get_accuracy(pos, true):
    reward = pos & true
    penalty = pos & ~true

    accuracy = float((reward.sum() - penalty.sum()) / true.sum())
    accuracy = round(accuracy, 4)
    print(
        f"Accuracy of this run: {accuracy},\n Reward: {reward.sum()} \n Penalty: {penalty.sum()}"
    )

    pass
