import torch

class MyTask:
    def __init__(self, cfg, model, train_dl, test_dl):
        self.model = model
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    def train_loop(self):
        size = len(self.train_dl.dataset)
        for batch, (X, y) in enumerate(self.train_dl):
            # Compute prediction and loss
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # print every 100 batch for info
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(self):
        size = len(self.test_dl.dataset)
        num_batches = len(self.test_dl)
        test_loss, correct = 0, 0

        with torch.no_grad():  # not learning, weights aren't changing
            for X, y in self.test_dl:
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def train(self, epochs):
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train_loop()
            self.test_loop()
        print('Training done')



