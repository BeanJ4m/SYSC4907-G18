import os
import glob
import pickle

import torch
import torch.nn as nn

from data_preparation import Dataloaders

def performance_testing(model, device, models_dir, results_dir):

    # Define the directory and file pattern
    directory = models_dir
    pattern = "tabtransformer_split_*.pt"

    # Find all matching files
    files = glob.glob(os.path.join(directory, pattern))

    # Extract numbers from file names
    numbers = []
    for file in files:
        base_name = os.path.basename(file)
        num_str = base_name.replace("tabtransformer_split_", "").replace(".pt", "")
        try:
            numbers.append(int(num_str))
        except ValueError:
            pass

    # Determine the maximum number
    max_num = max(numbers) if numbers else 0
    print(max_num)

    # Use the max_num in a loop
    for num in range(1, max_num + 1):
        file_path = os.path.join(models_dir, f"tabtransformer_split_{num}.pt")
        if os.path.exists(file_path):
            # Load the file or perform any operation you need
            print(f"Loading {file_path}")
        else:
            print(f"File {file_path} does not exist")

    pred_test = {}
    actual_test = {}
    accuracy_test = {}
    loss_test = {}
    G = 0

    for num in range(1, max_num+1):
        model_path = os.path.join(models_dir, f"tabtransformer_split_{num}.pt")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        prediction_matrix = []
        actual_matrix= []
        acc_matrix = []
        loss_matrix=[]
        G = G + 1
        criterion = nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in Dataloaders['Test']:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                prediction_matrix.append(predicted.tolist())
                actual_matrix.append(labels.tolist())
        loss /= len(Dataloaders['Test'].dataset)
        accuracy = correct / total
        loss_matrix.append(loss)
        acc_matrix.append(accuracy) 

        pred_test[f'Global_{G}'] = prediction_matrix
        actual_test[f'Global_{G}'] = actual_matrix
        accuracy_test[f'Global_{G}'] = acc_matrix
        loss_test[f'Global_{G}'] = loss_matrix 

        filename = os.path.join(results_dir, f'Global_{G}_pred')
        outfile = open(filename,'wb')
        pickle.dump(pred_test[f'Global_{G}'],outfile)
        outfile.close()

        filename = os.path.join(results_dir, f'Global_{G}_actual')
        outfile = open(filename,'wb')
        pickle.dump(actual_test[f'Global_{G}'],outfile)
        outfile.close()

        filename = os.path.join(results_dir, f'Global_{G}_accurracy')
        outfile = open(filename,'wb')
        pickle.dump(accuracy_test[f'Global_{G}'],outfile)
        outfile.close()

        filename = os.path.join(results_dir, f'Global_{G}_loss')
        outfile = open(filename,'wb')
        pickle.dump(loss_test[f'Global_{G}'],outfile)
        outfile.close()
