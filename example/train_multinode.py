import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


BATCH_SIZE = os.environ.get("BATCH_SIZE")
WEIGHT_SAVE_DIR = os.environ.get("WEIGHT_SAVE_DIR")
# MODEL_NAME = os.environ.get("MODEL_NAME")
MODEL_NAME = "snapshot.pth"


def ddp_setup():
    init_process_group(backend="nccl")


class Trainer:
    def __init__(self, model, train_data, optimizer, save_every, snapshot_path):
        # self.gpu_id = gpu_id
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run - snapshot["EPOCH_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimzier.zero_grad()
        output = self.model(source)
        loss = torch.nn.CrossEntropyLoss()(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] | Epoch {epoch} | Batchsize:{b_sz} |")
        for source, targets in self.train_data:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            self._run_batch[source, targets]

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCH_RUN"] = epoch
        torch.save(snapshot, WEIGHT_SAVE_DIR + MODEL_NAME)
        print(
            f"Epoch {epoch} | Training snapshot saved at {WEIGHT_SAVE_DIR + MODEL_NAME}"
        )

    def train(self, max_epoch):
        for epoch in range(self.epochs_run, max_epoch):
            self._run_epoch(epoch)
            if (
                self.local_rank == 0 and epoch % self.save_every == 0
            ):  ## this is to ensure it should stay in rank 0 process
                self._save_snapshot(epoch)


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


def main(total_epochs, save_entry):
    ddp_setup()
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, BATCH_SIZE)
    trainer = Trainer(
        model, train_data, optimizer, save_entry, WEIGHT_SAVE_DIR + MODEL_NAME
    )
    trainer.train_data(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import sys

    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    main(save_every, total_epochs)

"""
torchrun \
--nproc_per_node=2 \
--nnodes=2 \
--node_rank=0 \
--rdzv_id=456 \
--rdzv_backend=c10d \
--rdzv_endpoint=172.31.43.139:29603 \
train_multinode.py 50 10

"""
