import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset_loader import CarlaDataset
from model_bc import BCModel

# -------------------
# Config
# -------------------
DATA_PATH = "easycarla_offline_dataset.hdf5"
BATCH_SIZE = 1024
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------
# Dataset
# -------------------
dataset = CarlaDataset(DATA_PATH)

train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# -------------------
# Model
# -------------------
model = BCModel().to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_loss = float("inf")

# -------------------
# Training Loop
# -------------------
for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for states, actions in loop:
        states = states.to(DEVICE)
        actions = actions.to(DEVICE)

        pred = model(states)
        loss = criterion(pred, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    train_loss /= len(train_loader)

    # -------------------
    # Validation
    # -------------------
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for states, actions in val_loader:
            states = states.to(DEVICE)
            actions = actions.to(DEVICE)

            pred = model(states)
            loss = criterion(pred, actions)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"\nEpoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "bc_model_best.pth")
        print("✅ Saved Best Model")

print("Training Complete.")
