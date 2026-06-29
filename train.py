def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)

@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    """validation/test 공용."""
    model.eval()
    running_loss = 0.0
    y_true_all = []
    y_prob_all = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)

        running_loss += loss.item() * images.size(0)
        y_true_all.append(labels.cpu().numpy())
        y_prob_all.append(probs.cpu().numpy())

    loss = running_loss / len(loader.dataset)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    return loss, y_true_all, y_prob_all