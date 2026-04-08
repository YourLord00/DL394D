import os
import torch
import torch.nn as nn
from models import Classifier, save_model
from datasets.classification_dataset import load_data
from metrics import AccuracyMetric

if __name__ == "__main__":
    device = torch.device("mps")
    model = Classifier().to(device)
    train_loader = load_data("classification_data/train", transform_pipeline="aug", shuffle=True, num_workers=4)
    val_loader = load_data("classification_data/val", transform_pipeline="default", shuffle=False, num_workers=4)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_func(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        metric = AccuracyMetric()
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model.predict(images)
                metric.add(pred, labels)

        print(f"Epoch {epoch}, loss: {loss.item():.4f}, acc: {metric.compute()['accuracy']:.4f}")
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            os.makedirs("ckpts/classifier", exist_ok=True)
            torch.save(model.state_dict(), f"ckpts/classifier/epoch{epoch+1}.th")

    save_model(model)
