import torch
import numpy as np
from torch.utils.data import DataLoader


def test(dataset, model, batch_size):
    # TODO initialize a DataLoader on the dataset with the appropriate batch
    # size and shuffling enabled.
    data_loader =  DataLoader(dataset, batch_size=batch_size, shuffle=True)


    correct_count = 0
    for images, labels in data_loader:
        output = model.classify(images.cuda().view(batch_size, 1024))

        # Calculate accuracy
        _, predictions = torch.max(output, 1)

        # TODO calculate the number of correctly classified inputs.
        num_correct = (predictions.cpu() == labels).type(torch.float).sum().item()

        correct_count+=num_correct

    # TODO calculate the float accuracy for the whole dataset.
    accuracy = correct_count / len(data_loader.dataset)
    print("Testing Accuracy: %.3f"%(accuracy))
