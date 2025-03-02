import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import argparse
import torch.multiprocessing as mp


class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        is_ddp: bool = False,
        snapshot_path: str = None,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.is_ddp = is_ddp
        self.snapshot_path = snapshot_path
        self.epochs_run = 0

        if self.is_ddp:
            self.model = DDP(self.model, device_ids=[gpu_id])

        if self.snapshot_path and os.path.exists(self.snapshot_path):
            self._load_snapshot(self.snapshot_path)

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)

        # Handle DDP model state dict
        if self.is_ddp:
            # Remove 'module.' prefix from keys if present
            state_dict = {
                k.replace("module.", ""): v for k, v in snapshot["MODEL_STATE"].items()
            }
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(snapshot["MODEL_STATE"])

        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}"
        )
        if self.is_ddp:
            self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch):
        # Handle DDP model state dict
        if self.is_ddp:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        snapshot = {
            "MODEL_STATE": model_state,
            "EPOCHS_RUN": epoch,
        }
        torch.save(
            snapshot, self.snapshot_path if self.snapshot_path else "checkpoint.pt"
        )
        print(
            f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path if self.snapshot_path else 'checkpoint.pt'}"
        )

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int, is_ddp: bool = False):
    if is_ddp:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(dataset),
        )
    else:
        return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=True)


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank: int, world_size: int, args):
    if args.mode == "multigpu":
        ddp_setup(rank, world_size)
    elif args.mode == "multigpu_torchrun" or args.mode == "multinode":
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError(
                "LOCAL_RANK not found in environment variables. Use torchrun to launch this script."
            )
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        init_process_group(backend="nccl")

    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(
        dataset, args.batch_size, is_ddp=(args.mode != "single_gpu")
    )
    trainer = Trainer(
        model,
        train_data,
        optimizer,
        rank if args.mode != "single_gpu" else 0,
        args.save_every,
        is_ddp=(args.mode != "single_gpu"),
        snapshot_path=(
            "snapshot.pt"
            if args.mode == "multigpu_torchrun" or args.mode == "multinode"
            else None
        ),
    )
    trainer.train(args.total_epochs)

    if args.mode != "single_gpu":
        destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument(
        "mode",
        type=str,
        choices=["single_gpu", "multigpu", "multigpu_torchrun", "multinode"],
        help="Training mode",
    )
    parser.add_argument(
        "total_epochs", type=int, help="Total epochs to train the model"
    )
    parser.add_argument("save_every", type=int, help="How often to save a snapshot")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Input batch size on each device (default: 32)",
    )
    args = parser.parse_args()

    if args.mode == "single_gpu":
        main(0, 1, args)
    elif args.mode == "multigpu":
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size, args), nprocs=world_size)
    elif args.mode == "multigpu_torchrun" or args.mode == "multinode":
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError(
                "LOCAL_RANK not found in environment variables. Use torchrun to launch this script."
            )
        main(None, None, args)

# !python train_dp_07.py single_gpu 100 10 --batch_size 32
# !python train_dp_07.py multigpu 100 10 --batch_size 32
# !torchrun --nproc_per_node=2 train_dp_07.py multigpu_torchrun 100 10 --batch_size 32
