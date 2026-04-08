import os
import torch
import torch.nn as nn
from models import Detector, save_model
from datasets.road_dataset import load_data
from metrics import DetectionMetric

if __name__ == "__main__":
    device = torch.device("mps")
    model = Detector().to(device)
    train_loader = load_data("drive_data/train", transform_pipeline="aug", shuffle=True, num_workers=4)
    val_loader = load_data("drive_data/val", transform_pipeline="default", shuffle=False, num_workers=4)

    seg_loss_func = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0, 5.0]).to(device))
    depth_loss_func = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    num_epochs = 50

    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            depth = batch["depth"].to(device)
            track = batch["track"].to(device)
            
            logits, pred_depth = model(images)
            loss = 6 * seg_loss_func(logits, track) + depth_loss_func(pred_depth, depth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        metric = DetectionMetric()
        with torch.inference_mode():
            for batch in val_loader:
                images = batch["image"].to(device)
                depth = batch["depth"].to(device)
                track = batch["track"].to(device)

                pred, pred_depth = model.predict(images)
                metric.add(pred, track, pred_depth, depth)

        metrics = metric.compute()
        print(f"Epoch {epoch}, loss: {loss.item():.4f}, acc: {metrics['accuracy']:.4f}, iou: {metrics['iou']:.4f}, depth_err: {metrics['abs_depth_error']:.4f}, tp_depth_err: {metrics['tp_depth_error']:.4f}")
        scheduler.step()

        if epoch % 10 == 0:
            os.makedirs("ckpts/detector1", exist_ok=True)
            torch.save(model.state_dict(), f"ckpts/detector1/epoch{epoch}.th")

    save_model(model)
