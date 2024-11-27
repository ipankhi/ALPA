import time
import torch
import hashlib
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from utils import *
from dataset import *

def hash_tensor(tensor):
    """Hash a PyTorch tensor to a hexadecimal string."""
    return hashlib.sha256(tensor.byte().numpy().tobytes()).hexdigest()

def train(args, model, optimizer, loss_fn, train_dls, valid_dls, train_class_counts, valid_class_counts, num):
    """Train and evaluate the model using k-fold cross-validation."""
    best_precision = -1
    if args.is_master:
        print("\nStarting Training and Validation")

    for fold, (train_dl, valid_dl) in enumerate(zip(train_dls, valid_dls)):
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        print(f"Starting fold {fold + 1}/{args.n_splits}")

        for epoch in range(args.epochs):
            if args.is_master:
                print(f"Train: [Fold {fold + 1}, Epoch {epoch + 1}/{args.epochs}]")

            start_time = time.time()
            model.train()

            # Training loop
            for img, label in train_dl:
                img, label = img.to(torch.device(args.gpu_ids)), label.to(torch.device(args.gpu_ids))
                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=True):
                    pred = model(img)
                    if torch.isnan(pred).any():
                        print("NaN detected in predictions, skipping this batch")
                        continue
                    
                    if args.criterion == "LDAM":
                        loss = loss_fn(train_class_counts, pred, label.float())
                    else:
                        loss = loss_fn(pred, label.float())

                scaler.scale(loss).backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients to avoid exploding gradients
                scaler.step(optimizer)
                scaler.update()

            # Validation loop
            pred_list, label_list = torch.Tensor([]), torch.Tensor([])
            test_loss = 0

            if args.is_master:
                print(f"Eval: [Epoch {epoch + 1}/{args.epochs}]")
            model.eval()

            for img, label in valid_dl:
                img, label = img.to(torch.device(args.gpu_ids)), label.to(torch.device(args.gpu_ids))
                with torch.no_grad():
                    pred = model(img)
                    
                    if args.criterion == "LDAM":
                        loss = loss_fn(valid_class_counts, pred, label.float())
                    else:
                        loss = loss_fn(pred, label.float())
                    
                    test_loss += loss.item()
                    pred_list = torch.cat([pred_list, pred.cpu()], dim=0)
                    label_list = torch.cat([label_list, label.cpu()], dim=0)

            # Metrics calculation
            precision, acc, auc_score = metrics(args, pred_list, label_list)

            # Save the best model based on precision
            if args.save_model and precision > best_precision and args.is_master:
                best_precision = precision
                save_model(args, model, idx=num)
                print("Model saved with improved precision")

            if args.is_master:
                elapsed_time = time.time() - start_time
                print(f"{elapsed_time:.2f} seconds - Accuracy: {acc:.4f}, Precision: {precision:.4f}, "
                      f"Loss: {test_loss:.4f}, AUC Score: {auc_score:.4f}")

