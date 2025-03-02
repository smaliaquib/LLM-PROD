import os
import torch
from torch.utils.data import Dataset, DataLoader

# from utils import MYTrainDataset
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

WEIGHT_SAVE_DIR = os.environ.get("WEIGHT_SAVE_DIR")
# MODEL_NAME = os.environ.get("MODEL_NAME")
MODEL_NAME = "sample.pth"


def ddp_setup(rank, worl_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(backend="nccl", rank=rank, world_size=worl_size)


class Trainer:
    def __init__(self, model, train_data, optimizer, gpu_id, save_every):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _run_batch(self, source, targets):
        self.optimzier.zero_grad()
        output = self.model(source)
        loss = torch.nn.CrossEntropyLoss()(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] | Epoch {epoch} | Batchsize:{b_sz} |")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch[source, targets]

    def _save_checkpoint(self, epoch):
        # ckp = self.mode.state_dict()
        ckp = self.model.module.state_dict()
        torch.save(ckp, WEIGHT_SAVE_DIR + MODEL_NAME)
        print(
            f"Epoch {epoch} | Training checkpoint saved at checkpoint {WEIGHT_SAVE_DIR + MODEL_NAME}"
        )

    def train(self, max_epoch):
        for epoch in range(max_epoch):
            self._run_epoch(epoch)
            # if epoch % self.save_every == 0:
            if (
                self.gpu_id == 0 and epoch % self.save_every == 0
            ):  ## this is to ensure it should stay in rank 0 process
                self._save_checkpoint(epoch)


def load_train_objs():
    train_set = MyTrainDataset(2048)
    model = torch.nn.Linear(20, 1)
    optimizer = torch.optimi.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )


def main(rank, world_size, total_epochs, save_entry):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, BATCH_SIZE)
    trainer = Trainer(model, train_data, optimizer, rank, save_entry)
    trainer.train_data(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import sys

    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, total_epochs, save_every), nprocs=world_size)
    # main(device, total_epochs, save_every)
