import matplotlib.pyplot as plt
import numpy as np
import os


# Ejemplo: si guardas losses en training
train_losses = np.load("results/train_losses.npy")
val_losses = np.load("results/val_losses.npy")


plt.figure()

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")

plt.title("Curva de entrenamiento")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

os.makedirs("results", exist_ok=True)
plt.savefig("results/training_curve.png")

plt.show()