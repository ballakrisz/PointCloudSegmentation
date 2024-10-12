from torch.utils.tensorboard import SummaryWriter

class logger():
    def __init__(self, run_name) -> None:
        self.run_name = run_name
        self.writer = SummaryWriter(run_name)
        
    def log(self, epoch, train_loss, train_acc, val_loss, val_acc):
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Accuracy/train', train_acc, epoch)
        self.writer.add_scalar('Loss/validation', val_loss, epoch)
        self.writer.add_scalar('Accuracy/validation', val_acc, epoch)
        
    def close(self):
        self.writer.close()