#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-10T17:19:00.911Z
"""

# <font color='White'>***Libraries and Constants***</font>
# ---
# --- 


import pprint
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
ROUNDS = 40 
SPLITS = 1
CLASS_COUNT = 6

# <font color='Green'>***Metrics and Functions***</font>
# ---
# --- 


### This revised code for W recall, W precision, and W F1-score, take the distribution of actual as a reference instead of taking the distribution of the predictions!

def calculate_accuracy(actual_values, predicted_values):
    """Fraction of correct predictions."""
    correct_predictions = sum(a == p for a, p in zip(actual_values, predicted_values))
    total_samples = len(actual_values)
    return correct_predictions / total_samples

def calculate_weighted_precision(actual_values, predicted_values, n_classes):
    """
    Weighted Precision by actual distribution.
    weight_i = (# of actual samples for class i) / total_samples
    precision_i = TP_i / (TP_i + FP_i)
    """
    total_samples = len(actual_values)
    
    # Initialize counts
    tp = [0] * n_classes
    fp = [0] * n_classes
    actual_count = [0] * n_classes  # # of actual samples for each class

    # Count TP, FP, and actual totals
    for a, p in zip(actual_values, predicted_values):
        actual_count[a] += 1
        if a == p:
            tp[a] += 1
        else:
            fp[p] += 1

    precision_sum = 0.0
    for i in range(n_classes):
        # If this class is never predicted, precision_i can be taken as 0
        denominator = tp[i] + fp[i]
        precision_i = tp[i] / denominator if denominator > 0 else 0.0
        
        # Weighted by the proportion of this class in the *actual* labels
        weight_i = actual_count[i] / total_samples
        
        precision_sum += precision_i * weight_i
    
    return precision_sum

def calculate_weighted_recall(actual_values, predicted_values, n_classes):
    """
    Weighted Recall by actual distribution.
    weight_i = (# of actual samples for class i) / total_samples
    recall_i = TP_i / (TP_i + FN_i)
    """
    total_samples = len(actual_values)
    
    # Initialize counts
    tp = [0] * n_classes
    fn = [0] * n_classes
    actual_count = [0] * n_classes

    # Count TP, FN, and actual totals
    for a, p in zip(actual_values, predicted_values):
        actual_count[a] += 1
        if a == p:
            tp[a] += 1
        else:
            # only an FN if the true label was 'a' but not predicted as 'a'
            fn[a] += 1

    recall_sum = 0.0
    for i in range(n_classes):
        denominator = tp[i] + fn[i]
        recall_i = tp[i] / denominator if denominator > 0 else 0.0
        
        # Weighted by the proportion of this class in the actual labels
        weight_i = actual_count[i] / total_samples
        
        recall_sum += recall_i * weight_i

    return recall_sum

def calculate_weighted_f1(actual_values, predicted_values, n_classes):
    """
    Computes the weighted F1 score:
      F1_i per class i, weighted by the proportion of actual samples for i.
    """
    total_samples = len(actual_values)
    
    # Initialize TP, FP, FN for each class
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes
    
    # Populate counts
    for a, p in zip(actual_values, predicted_values):
        if a == p:
            tp[a] += 1
        else:
            fp[p] += 1  # Predicted p but actual != p
            fn[a] += 1  # Actual a but predicted != a

    f1_sum = 0.0
    # We will weight by the number of samples *actually* belonging to each class
    # (tp[i] + fn[i]) is the support (i.e., total actual count for class i)
    
    for i in range(n_classes):
        # Compute precision_i and recall_i
        precision_i = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        recall_i    = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        
        # Compute F1_i
        if precision_i + recall_i > 0:
            f1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i)
        else:
            f1_i = 0.0
        
        # Weight = proportion of samples in class i among all samples
        weight_i = (tp[i] + fn[i]) / total_samples if total_samples > 0 else 0.0
        
        f1_sum += f1_i * weight_i
    
    # The sum of weights across all classes should be 1 if total_samples > 0.
    # Hence, this sum is already the weighted F1.
    return f1_sum

def calculate_macro_recall(actual_values, predicted_values):
    class_labels = list(range(CLASS_COUNT))
    recall_sum = 0
    for label in class_labels:
        true_positive = 0
        false_negative = 0
        for actual, predicted in zip(actual_values, predicted_values):
            if actual == label and predicted == label:
                true_positive += 1
            elif actual == label and predicted != label:
                false_negative += 1
        # Avoid division by zero
        if true_positive + false_negative > 0:
            recall_sum += true_positive / (true_positive + false_negative)
    macro_recall = recall_sum / len(class_labels)
    return macro_recall

def calculate_macro_precision(actual_values, predicted_values):
    class_labels = list(range(CLASS_COUNT))
    precision_sum = 0
    for label in class_labels:
        true_positive = 0
        false_positive = 0
        for actual, predicted in zip(actual_values, predicted_values):
            if actual == label and predicted == label:
                true_positive += 1
            elif actual != label and predicted == label:
                false_positive += 1
        # Avoid division by zero
        if true_positive + false_positive > 0:
            precision_sum += true_positive / (true_positive + false_positive)
    macro_precision = precision_sum / len(class_labels)
    return macro_precision

def update_values(values_list):
    return [value if value >= 0.5 else 0.5 for value in values_list]

def pad_list(values_list):
    while len(values_list) < ROUNDS:
        values_list.append(values_list[-1])
    return values_list

def print_classification_report(actual, predicted, class_labels=None):
    if class_labels is None:
        class_labels = sorted(set(actual) | set(predicted))
    report = classification_report(actual, predicted, target_names=[str(label) for label in class_labels])
    print("Classification Report:\n")
    print(report)

def save_classification_report(actual, predicted, file_path, class_labels=None):
    if class_labels is None:
        class_labels = sorted(set(actual) | set(predicted))
    report = classification_report(actual, predicted, target_names=[str(label) for label in class_labels])
    with open(file_path, 'w') as file:
        file.write("Classification Report:\n\n")
        file.write(report)
    print(f"Classification report saved to {file_path}")

# <font color='Orange'>***Confusion Matrix***</font>
# ---
# --- 


Label_names = ['Normal', 'DDoS_UDP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_HTTP', 'Password'] #, 'Vulnerability_scanner', 'SQL_injection']#, 'Uploading', 'Backdoor', 'Port_Scanning', 'XSS', 'Ransomware', 'MITM', 'OS_Fingerprinting']
Label_numbers = [0, 1, 2, 3, 4, 5] #, 6, 7]#, 8, 9, 10, 11, 12, 13, 14]

def plot_confusion_matrix_with_names(actual, predicted, label_numbers, label_names, title="Confusion Matrix"):
    """
    Plots a confusion matrix with label names instead of numbers.

    Args:
        actual (list): The list of actual class labels.
        predicted (list): The list of predicted class labels.
        label_numbers (list): The list of numeric label identifiers.
        label_names (list): The corresponding list of label names.
        title (str): The title of the confusion matrix plot.
    """
    # Map numeric labels to their names
    label_map = dict(zip(label_numbers, label_names))
    
    # Compute confusion matrix
    cm = confusion_matrix(actual, predicted, labels=label_numbers)
    
    # Replace numeric labels with names for the axes
    class_labels = [label_map[num] for num in label_numbers]
    
    # Create a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', xticklabels=class_labels, yticklabels=class_labels)
    
    # Add labels, title, and a color bar
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# plot_confusion_matrix_with_names(Data["Perfect Post-Detection"]["Actual"][160], Data["Perfect Post-Detection"]["Predictions"][160], 
#                                  Label_numbers, Label_names, title="Confusion Matrix")

# <font color='Red'>***Calculating and Saving Results***</font>
# ---
# --- 

def main():
    Data = {}
    Data["TabTransformer Sequential"] = {}
    Data["TabTransformer Sequential"]['Path'] = "Data2/Sequential_Results"

    for Usecase in Data:
        Data[Usecase]['Actual'] = {}
        Data[Usecase]['Predictions'] = {}
    pprint.pprint(Data)

    for Usecase in Data:
        for i in range(1, ROUNDS*SPLITS+1):
            try:
                filename = f"{Data[Usecase]['Path']}/Global_{i}_actual"
                with open(filename, 'rb') as file:
                    Actual = pickle.load(file)
                Data[Usecase]['Actual'][i] = [item for sublist in Actual for item in sublist]
                filename = f"{Data[Usecase]['Path']}/Global_{i}_pred"
                with open(filename, 'rb') as file:
                    Pred = pickle.load(file)
                Data[Usecase]['Predictions'][i] = [item for sublist in Pred for item in sublist]
            except FileNotFoundError:
                print(f"File not found: {filename}")
            except Exception as e:
                print(f"An error occurred while processing file {filename}: {e}")

    # from collections import Counter
    # value_counts = Counter(Data['Confidence Thresholding']['Predictions'][160])
    # for value, count in value_counts.items():
    #     print(f"Value: {value}, Count: {count}")

    Results = {}
    for Usecase in Data:
        Results[Usecase] = {}
    for Usecase in Results:
        Results[Usecase]['Accuracy'] = []
        Results[Usecase]['Recall'] = []
        Results[Usecase]['Precision'] = [] 
        Results[Usecase]['F1_Score'] = []
    pprint.pprint(Results)

    for Usecase in Results:
        for Round in range(1, ROUNDS*SPLITS+1):
            try:
                Results[Usecase]['Accuracy'].append(calculate_accuracy(Data[Usecase]['Actual'][Round], Data[Usecase]['Predictions'][Round]))
                Results[Usecase]['Precision'].append(calculate_weighted_precision(Data[Usecase]['Actual'][Round], Data[Usecase]['Predictions'][Round], CLASS_COUNT))
                Results[Usecase]['Recall'].append(calculate_weighted_recall(Data[Usecase]['Actual'][Round], Data[Usecase]['Predictions'][Round], CLASS_COUNT))
                Results[Usecase]['F1_Score'].append(calculate_weighted_f1(Data[Usecase]['Actual'][Round], Data[Usecase]['Predictions'][Round], CLASS_COUNT))
            except KeyError:
                print('Usecase:', Usecase, 'Round:', Round)
                continue

    # for Usecase in Results:
    #     Results[Usecase]['Accuracy'] = update_values(Results[Usecase]['Accuracy'])
    #     Results[Usecase]['Recall'] = update_values(Results[Usecase]['Recall'])
    #     Results[Usecase]['Precision'] = update_values(Results[Usecase]['Precision'])    
    #     Results[Usecase]['F1_Score'] = update_values(Results[Usecase]['F1_Score'])    

    # <font color='Light Blue'>***Save/ Load Calculated Results***</font>
    # ---
    # --- 


    with open("PD_Results.pkl", "wb") as file:
        pickle.dump(Results, file)

    # with open("PD_Results.pkl", "rb") as file:
    #     Results = pickle.load(file)

    # <font color='Green'>***Plotting Results***</font>
    # ---
    # --- 


    plt.style.use('ggplot')
    LineStyle = ['ro-', 'b*-', 'ks-', 'gh-', 'm<-', 'yp-', 'b*-', 'gh-', 'rH-', 'c+-', 'mx-', 'ro-', 'b*-', 'ks-', 'gh-', 'y<-']
    fig = plt.figure(figsize=(17.2, 13), dpi=450)
    axx = fig.add_subplot(1,1,1)
    plt.figure(dpi=1000)
    # plt.figure(figsize=(8, 6))

    x_axis = np.array(np.arange(1, ROUNDS+1, 1).tolist())
    x_axis = np.insert(x_axis, 0, 0)
    y_points = {}
    for Usecase in Data:
        y_points[Usecase] = np.array(Results[Usecase]['Accuracy'])
        y_points[Usecase] = np.insert(y_points[Usecase], 0, 0.5)
    index = 0
    for Usecase in Data:
        axx.plot(x_axis, y_points[Usecase],LineStyle[index], label = Usecase, linewidth=2.5,  markersize=10)
        index += 1
        
    axx.set_xlabel('Federated Learning Detection Rounds', fontdict={'fontsize': 36})
    axx.set_ylabel('Detection Accuracy', fontdict={'fontsize': 36})
    axx.set_xticks(np.arange(0, ROUNDS+1, 5).tolist()) 
    axx.set_yticks(np.arange(0.30, 1.05, 0.05).tolist())
    axx.legend(loc = 'lower right', prop={'size': 16})
    axx.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=True, labelsize=20, colors='black')
    axx.xaxis.set_ticks_position('both')
    axx.tick_params(axis='y', which='both', left=True, right=True, labelleft=True, labelright=True, labelsize=20, colors='black')
    axx.yaxis.set_ticks_position('both')
    axx.xaxis.label.set_color('black')
    axx.yaxis.label.set_color('black')
    fig.savefig(f"1_Accuracy.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig) 

    plt.style.use('ggplot')
    LineStyle = ['ro-', 'b*-', 'ks-', 'gh-', 'm<-', 'yp-', 'b*-', 'gh-', 'rH-', 'c+-', 'mx-', 'ro-', 'b*-', 'ks-', 'gh-', 'y<-']
    fig = plt.figure(figsize=(17.2, 13), dpi=450)
    axx = fig.add_subplot(1,1,1)
    plt.figure(dpi=1000)
    # plt.figure(figsize=(8, 6))

    x_axis = np.array(np.arange(1, ROUNDS+1, 1).tolist())
    x_axis = np.insert(x_axis, 0, 0)
    y_points = {}
    for Usecase in Data:
        y_points[Usecase] = np.array(Results[Usecase]['Recall'])
        y_points[Usecase] = np.insert(y_points[Usecase], 0, 0.5)

    index = 0
    for Usecase in Data:
        axx.plot(x_axis, y_points[Usecase],LineStyle[index], label = Usecase, linewidth=2.5,  markersize=10)
        index += 1
        
    axx.set_xlabel('Federated Learning Detection Rounds', fontdict={'fontsize': 36})
    axx.set_ylabel('Detection Accuracy', fontdict={'fontsize': 36})
    axx.set_xticks(np.arange(0, ROUNDS+1, 5).tolist()) 
    axx.set_yticks(np.arange(0.30, 1.05, 0.05).tolist())
    axx.legend(loc = 'lower right', prop={'size': 16})
    axx.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=True, labelsize=20, colors='black')
    axx.xaxis.set_ticks_position('both')
    axx.tick_params(axis='y', which='both', left=True, right=True, labelleft=True, labelright=True, labelsize=20, colors='black')
    axx.yaxis.set_ticks_position('both')
    axx.xaxis.label.set_color('black')
    axx.yaxis.label.set_color('black')
    fig.savefig(f"1_Wieghted_Recall.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig) 

    plt.style.use('ggplot')
    LineStyle = ['ro-', 'b*-', 'ks-', 'gh-', 'm<-', 'yp-', 'b*-', 'gh-', 'rH-', 'c+-', 'mx-', 'ro-', 'b*-', 'ks-', 'gh-', 'y<-']
    fig = plt.figure(figsize=(17.2, 13), dpi=450)
    axx = fig.add_subplot(1,1,1)
    plt.figure(dpi=1000)
    # plt.figure(figsize=(8, 6))

    x_axis = np.array(np.arange(1, ROUNDS+1, 1).tolist())
    x_axis = np.insert(x_axis, 0, 0)
    y_points = {}
    for Usecase in Data:
        y_points[Usecase] = np.array(Results[Usecase]['Precision_Updated'])
        y_points[Usecase] = np.insert(y_points[Usecase], 0, 0.5)

    index = 0
    for Usecase in Data:
        axx.plot(x_axis, y_points[Usecase],LineStyle[index], label = Usecase, linewidth=2.5,  markersize=10)
        index += 1
        
    axx.set_xlabel('Federated Learning Detection Rounds', fontdict={'fontsize': 36})
    axx.set_ylabel('Detection Accuracy', fontdict={'fontsize': 36})
    axx.set_xticks(np.arange(0, ROUNDS+1, 5).tolist()) 
    axx.set_yticks(np.arange(0.30, 1.05, 0.05).tolist())
    axx.legend(loc = 'lower right', prop={'size': 16})
    axx.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=True, labelsize=20, colors='black')
    axx.xaxis.set_ticks_position('both')
    axx.tick_params(axis='y', which='both', left=True, right=True, labelleft=True, labelright=True, labelsize=20, colors='black')
    axx.yaxis.set_ticks_position('both')
    axx.xaxis.label.set_color('black')
    axx.yaxis.label.set_color('black')
    fig.savefig(f"1_Wieghted_Precision.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig) 

    plt.style.use('ggplot')
    LineStyle = ['ro-', 'b*-', 'ks-', 'gh-', 'm<-', 'yp-', 'b*-', 'gh-', 'rH-', 'c+-', 'mx-', 'ro-', 'b*-', 'ks-', 'gh-', 'y<-']
    fig = plt.figure(figsize=(17.2, 13), dpi=450)
    axx = fig.add_subplot(1,1,1)
    plt.figure(dpi=1000)
    # plt.figure(figsize=(8, 6))

    x_axis = np.array(np.arange(1, ROUNDS+1, 1).tolist())
    x_axis = np.insert(x_axis, 0, 0)
    y_points = {}
    for Usecase in Data:
        y_points[Usecase] = np.array(Results[Usecase]['F1_Score'])
        y_points[Usecase] = np.insert(y_points[Usecase], 0, 0.5)

    index = 0
    for Usecase in Data:
        axx.plot(x_axis, y_points[Usecase],LineStyle[index], label = Usecase, linewidth=2.5,  markersize=10)
        index += 1
        
    axx.set_xlabel('Federated Learning Detection Rounds', fontdict={'fontsize': 36})
    axx.set_ylabel('Detection Accuracy', fontdict={'fontsize': 36})
    axx.set_xticks(np.arange(0, ROUNDS+1, 5).tolist()) 
    axx.set_yticks(np.arange(0.30, 1.05, 0.05).tolist())
    axx.legend(loc = 'lower right', prop={'size': 16})
    axx.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=True, labelsize=20, colors='black')
    axx.xaxis.set_ticks_position('both')
    axx.tick_params(axis='y', which='both', left=True, right=True, labelleft=True, labelright=True, labelsize=20, colors='black')
    axx.yaxis.set_ticks_position('both')
    axx.xaxis.label.set_color('black')
    axx.yaxis.label.set_color('black')
    fig.savefig(f"1_Wieghted_F1_Score.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()