# best_epoch.py

def find_best_epoch(filename="lr_5e-5_hl_3_2_from_10/valid_loss.txt"):
    min_loss = None
    best_epoch = None

    with open(filename, "r") as f:
        for epoch, line in enumerate(f, start=1):
            value = float(line.strip())

            if min_loss is None or value < min_loss:
                min_loss = value
                best_epoch = epoch - 1

    return best_epoch, min_loss


if __name__ == "__main__":
    epoch, loss = find_best_epoch("lr_5e-5_hl_3_2_from_10/valid_loss.txt")
    print(f"Best epoch: {epoch}")
    print(f"Minimum validation loss: {loss}")
