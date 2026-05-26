import torch
import numpy as np
import sklearn 
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, confusion_matrix

from src.datasets.dataset import ChangeDetectionDataset
from src.models.deeplabv3plus import get_deeplabv3plus


def evaluate():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ChangeDetectionDataset(
        ["data/tiles_par1", "data/tiles_par2", "data/tiles_par3"],
        ["data/labels_par1", "data/labels_par2", "data/labels_par3"]
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    model = get_deeplabv3plus(18).to(device)
    model.load_state_dict(torch.load("models/modelo_final.pth"))
    model.eval()

    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:

            x = x.to(device)

            y_pred = model(x)
            y_pred = torch.sigmoid(y_pred)

            preds.append(y_pred.cpu())
            targets.append(y)

    preds = torch.cat(preds).view(-1).numpy()
    targets = torch.cat(targets).view(-1).numpy()

    preds = (preds > 0.5).astype(np.uint8)

    print("Precision:", precision_score(targets, preds, zero_division=0))
    print("Recall:", recall_score(targets, preds, zero_division=0))
    print("F1:", f1_score(targets, preds, zero_division=0))
    print("IoU:", jaccard_score(targets, preds, zero_division=0))

    print("\nMatriz de confusión:")
    print(confusion_matrix(targets, preds))


if __name__ == "__main__":
    evaluate()