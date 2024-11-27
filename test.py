import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.dataloader import TestDataset  # Import CustomDataset from your module
from dataset.augmentation import get_aug
from utils import get_args_parser_aptos
from models import get_model

# Parse arguments
args = get_args_parser_aptos()

# Load dataset and DataLoader
csv_file = args.test_csv_path
img_dir = args.img_path
_, valid_aug = get_aug(args)

# Initialize dataset and dataloader
dataset = TestDataset(csv_file=csv_file, img_dir=img_dir, transform=valid_aug)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load the model
model = get_model(args)
model_path = f"./{args.model_path}/convnext_{args.img_size}_{args.batchsize}{args.store_name}.pt"
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()
model = model.to(torch.device("cpu"))

# Evaluation loop
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in data_loader:
        images, labels = images.to(torch.device("cpu")), labels.to(torch.device("cpu"))
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Decode labels and predictions
all_preds = dataset.label_encoder.inverse_transform(all_preds)
all_labels = dataset.label_encoder.inverse_transform(all_labels)

# Classification report
report = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)
print(classification_report(all_labels, all_preds))

# Save report to CSV
report_df = pd.DataFrame(report).transpose()
report_path = f"./{args.model_path}/convnext_{args.img_size}_{args.batchsize}{args.store_name}.csv"
report_df.to_csv(report_path)

# Class-wise accuracy
def calculate_classwise_accuracy(labels, preds):
    cm = confusion_matrix(labels, preds)
    return cm.diagonal() / cm.sum(axis=1)

classwise_accuracy = calculate_classwise_accuracy(all_labels, all_preds)

# Add class-wise accuracy to report
label_to_index = {label: index for index, label in enumerate(dataset.label_encoder.classes_)}
sorted_labels = report_df.index[:-3]  # Exclude summary rows
sorted_classwise_accuracy = {label: classwise_accuracy[label_to_index[label]] for label in sorted_labels}
report_df['classwise_accuracy'] = np.nan
for label in sorted_labels:
    report_df.at[label, 'classwise_accuracy'] = sorted_classwise_accuracy[label]

# Save updated report
report_df.to_csv(report_path)

# Print class-wise accuracies
label_to_class_name = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative DR'}
for i, accuracy in enumerate(classwise_accuracy):
    print(f"Accuracy of class {label_to_class_name[i]}: {accuracy * 100:.2f}%")
