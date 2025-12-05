import argparse
import torch
from torch_geometric.data import DataLoader
from models.hetero_gae import HeteroGAE
from src.graph_dataset_dataloader import load_graph_dataloaders  # uses get_graph_dataset internally
import torch.nn.functional as F
from torch.optim import Adam
import tqdm
import torch.nn as nn
from prettytable import PrettyTable
from typing import Dict, Any, List


def pass_args():
    parser = argparse.ArgumentParser(description="Train HeteroGAE (lightweight test options)")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader (small default for weak machines)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension for encoder')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent dimension for encoder')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--limit_batches', type=int, default=0, help='If >0, limit number of batches per epoch (for quick tests)')
    parser.add_argument('--device', type=str, default=None, help='Force device: "cpu" or "cuda" (default auto)')
    parser.add_argument('--dataset', type=str, default='l2d', choices=['l2d','nuplan'], help='Which dataset loader to pick')
    return parser.parse_args()

def check_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"VRAM: allocated={allocated:.2f} GB | reserved={reserved:.2f} GB")
    return allocated, reserved

def model_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Analyzes a PyTorch model (nn.Module) and prints a detailed summary
    including parameters per module, total parameters, and trainable status.

    Args:
        model: The PyTorch model instance (nn.Module).

    Returns:
        A dictionary containing the total, trainable, and non-trainable parameter counts.
    """
    print("\n" + "="*80)
    print(f"Model Summary: {model.__class__.__name__}")
    print("="*80)

    # Use PrettyTable for clean, formatted output
    table = PrettyTable()
    table.field_names = ["Module Name", "Layer Type", "Param Count", "Requires Grad"]
    table.align = "l"

    total_params = 0
    trainable_params = 0

    # Iterate through all named modules (layers) in the model
    for name, module in model.named_modules():
        if name == "":
            # Skip the top-level module itself, which is often just the container
            continue

        # Check if the module contains any parameters
        module_params = 0
        has_trainable_param = False

        # Get the immediate parameters of this module, not its submodules
        # We iterate over the module's own named parameters
        for param_name, param in module.named_parameters(recurse=False):
            param_count = param.numel()
            module_params += param_count
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
                has_trainable_param = True

        # Only add a row if the module actually has parameters (skip things like ReLU)
        if module_params > 0:
            layer_type = module.__class__.__name__
            # Formatting the number with commas for readability
            formatted_params = f"{module_params:,}"
            grad_status = "Yes" if has_trainable_param else "No"

            table.add_row([name, layer_type, formatted_params, grad_status])

    non_trainable_params = total_params - trainable_params

    # Print the detailed table
    print(table)
    print("-" * 80)

    # Print the aggregated statistics
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"Non-Trainable Params:  {non_trainable_params:,}")

    # Calculate and print the percentage of trainable parameters
    if total_params > 0:
        trainable_percent = (trainable_params / total_params) * 100
        print(f"Trainable % of Total:  {trainable_percent:.2f}%")

    print("="*80 + "\n")

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params
    }

def build_pos_neg_edges(data, device):
    """Return dict of positive edge_index and negative sampled edge_index per relation"""
    pos = {}
    neg = {}
    for et in data.edge_types:
        eidx = data[et].edge_index
        # if no edges, keep empty
        if eidx.numel() == 0:
            pos[et] = eidx
            neg[et] = torch.empty((2,0), dtype=torch.long, device=eidx.device)
            continue
        pos[et] = eidx.to(device)
        # negative sampling per relation: sample same num_neg as pos edges
        # create negative samples wrt total node counts of src/dst types
        src_type, _, dst_type = et
        N_src = data[src_type].x.shape[0] if 'x' in data[src_type] else 0
        N_dst = data[dst_type].x.shape[0] if 'x' in data[dst_type] else 0
        if N_src==0 or N_dst==0:
            neg[et] = torch.empty((2,0), dtype=torch.long, device=device)
            continue
        # naive negative: sample random pairs
        E = eidx.shape[1]
        src_neg = torch.randint(0, N_src, (E,), device=device)
        dst_neg = torch.randint(0, N_dst, (E,), device=device)
        neg[et] = torch.stack([src_neg, dst_neg], dim=0)
    return pos, neg


def train(args):
    # select device
    # normalize device name: accept 'gpu' as alias for 'cuda'
    if args.device is not None:
        dev_str = args.device.lower()
        if dev_str == 'gpu':
            dev_str = 'cuda'
        device = torch.device(dev_str)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        print("torch.cuda.device_count() ->", torch.cuda.device_count())
    except Exception:
        print("torch.cuda.device_count() -> unavailable")

    # load dataloaders (uses get_graph_dataset internally)
    l2d_loader, nuplan_loader = load_graph_dataloaders(shuffle=True, batch_size=args.batch_size)
    loader = l2d_loader if args.dataset == 'l2d' else nuplan_loader

    # inspect one batch to get node input dims & relation types
    try:
        batch = next(iter(loader))
    except StopIteration:
        raise RuntimeError('Selected dataloader is empty')

    in_dims = {}
    for n in batch.node_types:
        node_store = batch[n]
        if 'x' in node_store:
            in_dims[n] = node_store.x.shape[1]
    # relation types present in dataset
    relation_types = list(batch.edge_types)

    # build model
    model = HeteroGAE(in_dims, relation_types, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)
    model_summary(model)
    model = model.to(device)
    # Check model parameter device
    try:
        some_param = next(model.parameters())
        print("Model parameters on device:", some_param.device)
    except StopIteration:
        print("Model has no parameters to inspect")
    opt = Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches_processed = 0
        first_batch_done = False
        with tqdm.tqdm(total=len(loader), desc=f"Epoch {epoch}", unit="batch") as pbar:
            for batch in loader:
                allocated = torch.cuda.memory_allocated() / 1e9
                gpu_memory_status = f"{allocated:.2f}GB allocated"

                batch = batch.to(device)

                pos_edges, neg_edges = build_pos_neg_edges(batch, device)
                # Instrument first batch: time forward+backward and print devices
                if not first_batch_done:
                    import time
                    if device.type == 'cuda' and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    logits_pos, _ = model(batch, pos_edges)
                    logits_neg, _ = model(batch, neg_edges)
                    if device.type == 'cuda' and torch.cuda.is_available():
                        torch.cuda.synchronize()

                    first_batch_done = True
                else:
                    logits_pos, _ = model(batch, pos_edges)
                    logits_neg, _ = model(batch, neg_edges)
                # initialize loss as tensor on device so backward() works
                loss = torch.tensor(0.0, device=device)
                for et in batch.edge_types:
                    lp = logits_pos.get(et, torch.tensor([], device=device))
                    ln = logits_neg.get(et, torch.tensor([], device=device))
                    if lp.numel() == 0 and ln.numel() == 0:
                        continue
                    # BCE on positives and negatives
                    pos_loss = F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp))
                    neg_loss = F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln))
                    loss = loss + pos_loss + neg_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                batches_processed += 1
                
                pbar.update(1)
                # pbar.set_postfix(loss=f"{total_loss:.4f}")
                pbar.set_postfix_str(f"loss={total_loss:.4f} | {gpu_memory_status}")

                if args.limit_batches > 0 and batches_processed >= args.limit_batches:
                    break
        print(f"Epoch {epoch} loss: {total_loss:.4f} (batches: {batches_processed})")


if __name__ == "__main__":
    args = pass_args()
    train(args)
