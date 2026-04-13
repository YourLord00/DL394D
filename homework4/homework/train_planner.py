"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

print("Time to train")

import os
import argparse
from homework.models import MLPPlanner, TransformerPlanner, CNNPlanner, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='mlp_planner', choices=['mlp_planner', 'transformer_planner', 'cnn_planner'])
args = parser.parse_args()

if args.model == 'cnn_planner':
    transform = 'default'
else:
    transform = 'state_only'

train_loader = load_data('drive_data/train', transform_pipeline=transform, shuffle=True)
val_loader = load_data('drive_data/val', transform_pipeline=transform)

if args.model == 'mlp_planner':
    model = MLPPlanner()
    lr = 1e-3
    epochs = 50
elif args.model == 'transformer_planner':
    model = TransformerPlanner()
    lr = 1e-4
    epochs = 50
elif args.model == 'cnn_planner':
    model = CNNPlanner()
    lr = 1e-3
    epochs = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = model.to(device)

os.makedirs(f"homework/{args.model}_checkpoints", exist_ok=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

best_lat_err = float('inf')
best_long_err = float('inf')

for epoch in range(epochs):
    model.train()
    
    for batch in train_loader:
        waypoints = batch['waypoints'].to(device)
        waypoints_mask = batch['waypoints_mask'].to(device)

        if args.model == 'cnn_planner':
            prediction_waypoints = model(batch['image'].to(device))
        else:
            prediction_waypoints = model(batch['track_left'].to(device), batch['track_right'].to(device))
        loss = (prediction_waypoints - waypoints).abs()
        if args.model == 'cnn_planner':
            pass
        else:
            loss[:, :, 1] = loss[:, :, 1] * 2.0
        loss = loss * waypoints_mask[..., None]
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()


    model.eval()
    metric = PlannerMetric()
    with torch.no_grad():
        for batch in val_loader:
            if args.model == 'cnn_planner':
                pred = model(image=batch['image'].to(device))
            else:
                pred = model(track_left=batch['track_left'].to(device), track_right=batch['track_right'].to(device))
            metric.add(pred, batch['waypoints'].to(device), batch['waypoints_mask'].to(device))
    results = metric.compute()
    lat_err = results['lateral_error']
    print(f"Epoch {epoch} loss: {loss.item():.4f} | long_err: {results['longitudinal_error']:.4f} | lat_err: {lat_err:.4f}")

    long_err = results['longitudinal_error']

    if args.model == 'cnn_planner':
        if long_err < 0.3 and lat_err < 0.45:
            if (long_err < best_long_err and lat_err <= best_lat_err) \
            or (lat_err < best_lat_err and long_err <= best_long_err):
                best_long_err = min(best_long_err, long_err)
                best_lat_err = min(best_lat_err, lat_err)
                model.cpu()
                save_model(model)
                model.to(device)
                print(f"  -> saved best model (lat_err: {lat_err:.4f}, long_err: {long_err:.4f})")
    else:
        if lat_err < best_lat_err:
            best_lat_err = lat_err
            model.cpu()
            save_model(model)
            model.to(device)
            print(f"  -> saved best model (lat_err: {lat_err:.4f})")

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"homework/{args.model}_checkpoints/epoch_{epoch+1}.th")
        print(f"  -> checkpoint saved at epoch {epoch+1}")

print(f"Best lateral error: {best_lat_err:.4f}")
