"""
Training with Gumbel-Softmax Sampling and Dynamic Penalty Method.

Reference: L2O-DiffILO training methodology.
The GNN predicts logits for binary variables, solutions are sampled via
Gumbel-Softmax, and the loss combines the expected objective with a
constraint violation penalty modulated by a dynamically adjusted mu.

The neural network (GNNPolicy) is NOT modified.
"""

import argparse
import os
import time
import random
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.utils import unbatch

from utils import TASKS, extract_raw_ilp
from gnn import GNNPolicy
from dataset.unsupervised_dataset import UnsupervisedGraphDataset

os.environ['DGLBACKEND'] = "pytorch"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


TASK_BATCH_SIZE = {'CA': 4, 'WA': 4, 'IP': 4, 'SC': 1, 'IS': 4, '2club': 1}


# ============================================================
#  Gumbel-Softmax Sampling
# ============================================================

def gumbel_sample(logits, N, tau=1.0):
    """
    Gumbel-Softmax sampling for differentiable binary decisions.

    Args:
        logits: [n_vars] or [n_vars, 1] raw logits from the GNN.
        N: number of solutions to sample.
        tau: Gumbel-Softmax temperature.

    Returns:
        [N, n_vars] binary samples (hard via straight-through estimator).
    """
    logits = logits.reshape(-1, 1)
    logits = logits.repeat(N, 1, 1)                                  # [N, n_vars, 1]
    logits = torch.cat([torch.zeros_like(logits), logits], dim=-1)    # [N, n_vars, 2]
    return torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True)[:, :, 1]


# ============================================================
#  Per-graph Utility Functions
# ============================================================

def build_dense_A(raw_cons_indices, raw_cons_values, n_cons, n_vars, device):
    """Build dense constraint matrix A from sparse representation."""
    A = torch.zeros(n_cons, n_vars, device=device)
    row = raw_cons_indices[0].to(device)
    col = raw_cons_indices[1].to(device)
    val = raw_cons_values.to(device)
    A[row, col] = val
    return A


def map_logits_to_raw(logits_gnn, gnn_to_raw_map, n_raw_vars, device):
    """
    Map GNN-ordered logits to raw ILP variable order via scatter-average.
    Handles cases where multiple GNN outputs map to the same raw variable.
    """
    logits_raw = torch.zeros(n_raw_vars, device=device)
    count = torch.zeros(n_raw_vars, device=device)
    gnn_to_raw = gnn_to_raw_map.to(device)
    logits_raw.scatter_add_(0, gnn_to_raw, logits_gnn)
    count.scatter_add_(0, gnn_to_raw, torch.ones_like(logits_gnn))
    count = count.clamp(min=1)
    return logits_raw / count


def graph_is_minimize(graph):
    if hasattr(graph, 'obj_sense_min'):
        obj_sense = graph.obj_sense_min
        if torch.is_tensor(obj_sense):
            return bool(obj_sense.reshape(-1)[0].item())
        return bool(obj_sense)
    return True


def initial_best_obj(is_minimize):
    return float('inf') if is_minimize else float('-inf')


def objective_is_better(candidate, best, is_minimize):
    return candidate < best if is_minimize else candidate > best


# ============================================================
#  Loss Computation (per graph)
# ============================================================

def compute_loss_per_graph(logits_gnn, graph, num_samples, mu, loss_config, device):
    """
    Compute the Gumbel-Softmax penalty loss for a single graph.

    Args:
        logits_gnn: [n_gnn_vars] raw logits in GNN variable order.
        graph: individual graph data (un-batched).
        num_samples: number of Gumbel-Softmax samples.
        mu: penalty coefficient for constraint violations.
        loss_config: one of 'normalize', 'mean', 'sum', 'nonzero_mean'.
        device: torch device.

    Returns:
        loss: scalar loss for this graph.
        obj: scalar expected objective value.
        cons_sum: scalar sum of constraint violations.
    """
    n_raw_vars = graph.obj_coeffs.shape[0]
    n_cons = graph.raw_n_cons if isinstance(graph.raw_n_cons, int) else graph.raw_n_cons.item()

    # Map GNN logits to raw ILP variable order
    logits_raw = map_logits_to_raw(logits_gnn, graph.gnn_to_raw_map, n_raw_vars, device)

    # Build dense A and get b, c
    A = build_dense_A(graph.raw_cons_indices, graph.raw_cons_values, n_cons, n_raw_vars, device)
    b = graph.raw_rhs.to(device).reshape(-1, 1)
    c = graph.obj_coeffs.to(device).reshape(-1, 1)

    # Gumbel-Softmax sample discrete solutions
    x = gumbel_sample(logits_raw, num_samples, tau=1.0).float().reshape(num_samples, -1)

    # Expected objective: p^T c
    p = torch.sigmoid(logits_raw).reshape(-1, 1)
    obj = (p * c).sum()

    # Constraint violation: ReLU(A x - b), averaged over samples
    cons_pos = torch.relu(A @ x.T - b).mean(dim=1, keepdim=True)  # [n_cons, 1]

    # Loss according to config
    if loss_config == "normalize" and torch.norm(c) > 0:
        num_nonzero = torch.count_nonzero(cons_pos)
        if num_nonzero > 0:
            A_row_norm = torch.norm(A, dim=1).clamp(min=1e-8)
            loss = obj / torch.norm(c) + mu * (cons_pos.squeeze() / A_row_norm).sum() / num_nonzero
        else:
            loss = obj / torch.norm(c)
    elif loss_config == "sum":
        loss = obj + mu * cons_pos.sum()
    elif loss_config == "nonzero_mean":
        num_nonzero = torch.count_nonzero(cons_pos)
        if num_nonzero > 0:
            loss = obj + mu * cons_pos.sum() / num_nonzero
        else:
            loss = obj
    else:  # "mean" (default)
        loss = obj + mu * cons_pos.mean()

    return loss, obj.detach(), cons_pos.sum().detach()


# ============================================================
#  Validation Sampling
# ============================================================

@torch.no_grad()
def evaluate_by_sampling(logits_gnn, graph, n_eval_samples, device):
    """
    Evaluate a graph by sampling many solutions and finding the best feasible one.

    Returns:
        best_feasible_obj: best objective among feasible solutions.
        best_obj: best objective among all solutions.
        mean_obj: mean objective over all solutions.
        mean_feasible_obj: mean objective among feasible solutions.
        n_feasible: number of feasible solutions found.
        best_feasible_solution: best feasible binary solution, or None.
    """
    n_raw_vars = graph.obj_coeffs.shape[0]
    n_cons = graph.raw_n_cons if isinstance(graph.raw_n_cons, int) else graph.raw_n_cons.item()
    is_minimize = graph_is_minimize(graph)

    logits_raw = map_logits_to_raw(logits_gnn, graph.gnn_to_raw_map, n_raw_vars, device)

    A = build_dense_A(graph.raw_cons_indices, graph.raw_cons_values, n_cons, n_raw_vars, device)
    b = graph.raw_rhs.to(device).reshape(-1, 1)
    c = graph.obj_coeffs.to(device).reshape(-1, 1)

    xx = gumbel_sample(logits_raw, n_eval_samples, tau=1.0).float().reshape(n_eval_samples, -1)
    objs = (xx @ c).squeeze(-1)
    violations = torch.relu(A @ xx.T - b).sum(dim=0)
    feasible_mask = (violations == 0)
    n_feasible = feasible_mask.sum().item()

    if n_feasible > 0:
        feasible_objs = objs[feasible_mask]
        feasible_solutions = xx[feasible_mask]
        best_idx = torch.argmin(feasible_objs) if is_minimize else torch.argmax(feasible_objs)
        best_feasible_obj = feasible_objs[best_idx].item()
        best_feasible_solution = feasible_solutions[best_idx].detach().cpu()
        mean_feasible_obj = feasible_objs.mean().item()
    else:
        best_feasible_obj = initial_best_obj(is_minimize)
        best_feasible_solution = None
        mean_feasible_obj = initial_best_obj(is_minimize)

    best_obj_idx = torch.argmin(objs) if is_minimize else torch.argmax(objs)
    best_obj = objs[best_obj_idx].item()
    mean_obj = objs.mean().item()

    return best_feasible_obj, best_obj, mean_obj, mean_feasible_obj, n_feasible, best_feasible_solution


@torch.no_grad()
def evaluate_by_rounding(logits_gnn, graph, device):
    """
    Evaluate a graph by rounding sigmoid(logits) to 0/1 and checking feasibility.

    Returns:
        is_feasible: whether the rounded solution satisfies all constraints.
        obj_val: objective value of the rounded solution.
        solution: rounded binary solution.
    """
    n_raw_vars = graph.obj_coeffs.shape[0]
    n_cons = graph.raw_n_cons if isinstance(graph.raw_n_cons, int) else graph.raw_n_cons.item()

    logits_raw = map_logits_to_raw(logits_gnn, graph.gnn_to_raw_map, n_raw_vars, device)

    A = build_dense_A(graph.raw_cons_indices, graph.raw_cons_values, n_cons, n_raw_vars, device)
    b = graph.raw_rhs.to(device).reshape(-1, 1)
    c = graph.obj_coeffs.to(device).reshape(-1, 1)

    x_round = torch.round(torch.sigmoid(logits_raw)).reshape(1, -1)
    obj_val = (x_round @ c).item()
    violation = torch.relu(A @ x_round.T - b).sum().item()
    is_feasible = (violation == 0)

    return is_feasible, obj_val, x_round.reshape(-1).detach().cpu()


# ============================================================
#  Model Forward Pass Helper
# ============================================================

def model_forward(model, batch, device):
    """
    Run the GNN forward pass on a batch, constructing the necessary
    batch assignment tensors.

    Returns:
        logits: [total_vars] raw logits (before sigmoid).
        variable_features_batch: [total_vars] batch assignment for variables.
    """
    constraint_features_batch = torch.repeat_interleave(
        torch.arange(len(batch.ntcons), device=device),
        batch.ntcons.clone().detach().long()
    )
    variable_features_batch = torch.repeat_interleave(
        torch.arange(len(batch.ntvars), device=device),
        batch.ntvars.clone().detach().long()
    )

    batch.constraint_features[torch.isinf(batch.constraint_features)] = 10

    logits = model(
        batch.constraint_features,
        batch.edge_index,
        batch.edge_attr,
        batch.variable_features,
        batch.n_constraints,
        constraint_features_batch,
        variable_features_batch,
    )

    return logits, variable_features_batch


# ============================================================
#  Training Epoch
# ============================================================

def train_epoch(model, data_loader, optimizer, mu, num_samples,
                loss_config, grad_clip_norm, device):
    """
    One epoch of training with Gumbel-Softmax sampling.

    Returns:
        epoch_loss, epoch_obj, epoch_cons, num_graphs
    """
    model.train()
    epoch_loss = 0.0
    epoch_obj = 0.0
    epoch_cons = 0.0
    num_graphs = 0

    for batch in data_loader:
        batch = batch.to(device)

        # Forward pass
        logits, var_batch = model_forward(model, batch, device)

        # Split logits per graph
        logits_per_graph = unbatch(logits, var_batch)
        graphs = batch.to_data_list()

        batch_loss = torch.zeros(1, device=device)
        batch_obj = 0.0
        batch_cons = 0.0

        for i, g in enumerate(graphs):
            loss_i, obj_i, cons_i = compute_loss_per_graph(
                logits_per_graph[i], g, num_samples, mu, loss_config, device
            )
            batch_loss += loss_i
            batch_obj += obj_i.item()
            batch_cons += cons_i.item()

        n_graphs = len(graphs)
        batch_loss = batch_loss / n_graphs

        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        epoch_loss += batch_loss.item() * n_graphs
        epoch_obj += batch_obj
        epoch_cons += batch_cons
        num_graphs += n_graphs

    return epoch_loss, epoch_obj, epoch_cons, num_graphs


# ============================================================
#  Validation Epoch
# ============================================================

@torch.no_grad()
def valid_epoch(model, data_loader, mu, num_samples, loss_config,
                n_eval_samples, device):
    """
    Validation: compute loss + sampling-based evaluation + rounding-based evaluation.

    Returns dict with: loss, obj, cons, best_feasible, best_obj, mean_obj,
                       mean_feasible_obj, total_feasible, total_samples,
                       round_n_feasible, round_avg_obj, n_valid_instances
    """
    model.eval()
    epoch_loss = 0.0
    epoch_obj = 0.0
    epoch_cons = 0.0
    epoch_best_feasible = 0.0       # only feasible instances
    epoch_best_feasible_all = 0.0   # all instances (inf if any infeasible)
    epoch_best_obj = 0.0
    epoch_mean_obj = 0.0
    epoch_mean_feasible_obj = 0.0
    epoch_total_feasible = 0
    num_graphs = 0

    # Rounding-based metrics
    round_n_feasible = 0
    round_obj_sum_feasible = 0.0
    # Count instances where sampling found at least one feasible solution
    n_sample_feasible_instances = 0

    for batch in data_loader:
        batch = batch.to(device)

        logits, var_batch = model_forward(model, batch, device)
        logits_per_graph = unbatch(logits, var_batch)
        graphs = batch.to_data_list()

        for i, g in enumerate(graphs):
            loss_i, obj_i, cons_i = compute_loss_per_graph(
                logits_per_graph[i], g, num_samples, mu, loss_config, device
            )
            best_feas, best_obj, mean_obj, mean_feas_obj, n_feas, _ = evaluate_by_sampling(
                logits_per_graph[i], g, n_eval_samples, device
            )
            is_round_feasible, round_obj, _ = evaluate_by_rounding(
                logits_per_graph[i], g, device
            )

            epoch_loss += loss_i.item()
            epoch_obj += obj_i.item()
            epoch_cons += cons_i.item()
            epoch_best_feasible_all += best_feas
            epoch_best_obj += best_obj
            epoch_mean_obj += mean_obj
            epoch_total_feasible += n_feas
            num_graphs += 1

            # Sampling: only accumulate over feasible instances
            if n_feas > 0:
                epoch_best_feasible += best_feas
                epoch_mean_feasible_obj += mean_feas_obj
                n_sample_feasible_instances += 1

            # Rounding metrics
            if is_round_feasible:
                round_n_feasible += 1
                round_obj_sum_feasible += round_obj

    num_graphs = max(num_graphs, 1)
    return {
        'loss': epoch_loss / num_graphs,
        'obj': epoch_obj / num_graphs,
        'cons': epoch_cons / num_graphs,
        'best_feasible': epoch_best_feasible / max(n_sample_feasible_instances, 1),
        'avg_best_feas_all': epoch_best_feasible_all / num_graphs,
        'best_obj': epoch_best_obj / num_graphs,
        'mean_obj': epoch_mean_obj / num_graphs,
        'mean_feasible_obj': epoch_mean_feasible_obj / max(n_sample_feasible_instances, 1),
        'total_feasible': epoch_total_feasible,
        'n_eval_samples': n_eval_samples * num_graphs,
        'n_sample_feasible_instances': n_sample_feasible_instances,
        'round_n_feasible': round_n_feasible,
        'round_avg_obj': round_obj_sum_feasible / max(round_n_feasible, 1),
        'n_valid_instances': num_graphs,
    }


# ============================================================
#  Argument Parser
# ============================================================

def get_parser():
    parser = argparse.ArgumentParser(
        description="GNN Training for ILP with Gumbel-Softmax Sampling."
    )

    # Problem
    parser.add_argument("--problem_type", choices=TASKS, default='SC')

    # Model architecture (unchanged)
    parser.add_argument("--gnn_type", default='gcn')
    parser.add_argument("--emb_size", type=int, default=64)
    parser.add_argument("--cons_nfeats", type=int, default=4)
    parser.add_argument("--edge_nfeats", type=int, default=1)
    parser.add_argument("--var_nfeats", type=int, default=6)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument('--Intra_Constraint_Competitive', default=False,
                        action='store_true')

    # Training hyperparameters
    parser.add_argument("--lr_output", type=float, default=5e-4,
                        help="Learning rate for output layers (default: %(default)s)")
    parser.add_argument("--lr_inner", type=float, default=5e-4,
                        help="Learning rate for GNN body (default: %(default)s)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="L2 regularization (default: %(default)s)")
    parser.add_argument("--num_epochs", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override task-specific batch size")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of Gumbel-Softmax samples per instance (default: %(default)s)")
    parser.add_argument("--n_eval_samples", type=int, default=30,
                        help="Number of samples for validation evaluation (default: %(default)s)")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping, 0=disabled (default: %(default)s)")

    # Loss configuration
    parser.add_argument("--loss_config", type=str, default='normalize',
                        choices=['normalize', 'mean', 'sum', 'nonzero_mean'],
                        help="Loss configuration (default: %(default)s)")

    # Dynamic penalty mu
    parser.add_argument("--mu_init", type=float, default=0.3,
                        help="Initial penalty coefficient (default: %(default)s)")
    parser.add_argument("--mu_step_size", type=float, default=0.01,
                        help="Step size for mu update (default: %(default)s)")
    parser.add_argument("--mu_value", type=float, default=1.0,
                        help="Target constraint violation for mu update (default: %(default)s)")
    parser.add_argument("--mu_max", type=float, default=5.0,
                        help="Maximum mu value (default: %(default)s)")
    parser.add_argument("--mu_min", type=float, default=0.01,
                        help="Minimum mu value (default: %(default)s)")

    # LR schedule
    parser.add_argument("--lr_schedule", choices=['cos', 'cosrestart', 'exp', 'none'],
                        default='exp',
                        help="LR schedule type (default: %(default)s)")
    parser.add_argument("--cos_T", type=int, default=200,
                        help="T_max for CosineAnnealingLR (default: %(default)s)")
    parser.add_argument("--cos_min", type=float, default=0.0,
                        help="Minimum LR for cosine schedule (default: %(default)s)")
    parser.add_argument("--lr_anneal_factor", type=float, default=0.88,
                        help="Factor for ExponentialLR (default: %(default)s)")

    # Paths
    parser.add_argument("--instance_dir",
                        default="/home/lmh/autodl-tmp/data/l2o_milp",
                        help="Directory containing .lp/.mps instance files")
    parser.add_argument("--cache_dir", default=None,
                        help="Cache directory for preprocessed data")
    parser.add_argument("--model_save_dir", default="./pretrain_models")
    parser.add_argument("--log_save_dir", default="./train_logs")
    parser.add_argument("--solution_save_dir", default="./solution_records",
                        help="Directory for per-instance best solution files")
    parser.add_argument("--tensorboard_dir", default="./tb_logs",
                        help="TensorBoard log directory")
    parser.add_argument("--record_artifacts", default=False, action='store_true',
                        help="Record model weights, log files, and TensorBoard data (default: disabled)")

    # Resume
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to .pth checkpoint to resume training from.")

    # Device
    parser.add_argument("--device", default="cuda:0")

    # Validation frequency
    parser.add_argument("--val_every", type=int, default=1,
                        help="Validate every N epochs (default: %(default)s)")

    # Patience
    parser.add_argument("--patience", type=int, default=200,
                        help="Early-stop patience (default: %(default)s)")

    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: %(default)s)")

    return parser


# ============================================================
#  Main helpers
# ============================================================

def build_model(args, device):
    return GNNPolicy(
        emb_size=args.emb_size,
        cons_nfeats=args.cons_nfeats,
        edge_nfeats=args.edge_nfeats,
        var_nfeats=args.var_nfeats,
        depth=args.depth,
        Intra_Constraint_Competitive=args.Intra_Constraint_Competitive,
    ).to(device)


def build_optimizer(model, args):
    output_param_ids = set()
    for layer in (model.vars_output_layer, model.cons_output_layer):
        for p in layer.parameters():
            output_param_ids.add(id(p))
    other_params = [p for p in model.parameters() if id(p) not in output_param_ids]

    return optim.Adam([
        {'params': model.vars_output_layer.parameters(), 'lr': args.lr_output},
        {'params': model.cons_output_layer.parameters(), 'lr': args.lr_output},
        {'params': other_params, 'lr': args.lr_inner},
    ], weight_decay=args.weight_decay)


def build_scheduler(optimizer, args):
    if args.lr_schedule == 'cos':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.cos_T, eta_min=args.cos_min
        )
    if args.lr_schedule == 'cosrestart':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.cos_T, T_mult=2, eta_min=args.cos_min
        )
    if args.lr_schedule == 'exp':
        return optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_anneal_factor
        )
    return None


@torch.no_grad()
def evaluate_current_instance(model, data_loader, n_eval_samples, device):
    model.eval()
    batch = next(iter(data_loader)).to(device)
    logits, var_batch = model_forward(model, batch, device)
    logits_per_graph = unbatch(logits, var_batch)
    graph = batch.to_data_list()[0]

    round_feasible, round_obj, round_solution = evaluate_by_rounding(
        logits_per_graph[0], graph, device
    )
    sample_obj, _, _, _, sample_n_feasible, sample_solution = evaluate_by_sampling(
        logits_per_graph[0], graph, n_eval_samples, device
    )

    return {
        'is_minimize': graph_is_minimize(graph),
        'round_feasible': round_feasible,
        'round_obj': round_obj,
        'round_solution': round_solution,
        'sample_obj': sample_obj,
        'sample_n_feasible': sample_n_feasible,
        'sample_solution': sample_solution,
    }


def count_selected(solution):
    if solution is None:
        return 0
    return int(solution.reshape(-1).sum().item())


def format_best_result(result):
    if result is None:
        return "None"
    return (
        f"obj={result['obj']:.6g}, epoch={result['epoch']}, "
        f"time={result['time']:.3f}s, n_vars={result['solution'].numel()}, "
        f"n_ones={count_selected(result['solution'])}"
    )


def safe_instance_stem(instance_path):
    stem = os.path.splitext(os.path.basename(instance_path))[0]
    return ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in stem)


def write_line(log_file, text):
    if log_file is not None:
        log_file.write(text + '\n')
        log_file.flush()


def best_overall_result(best_round, best_sample, is_minimize):
    if best_round is None:
        if best_sample is None:
            return None
        return {'method': 'sample', **best_sample}
    if best_sample is None:
        return {'method': 'round', **best_round}
    if objective_is_better(best_sample['obj'], best_round['obj'], is_minimize):
        return {'method': 'sample', **best_sample}
    return {'method': 'round', **best_round}


def solution_vector_text(solution):
    if solution is None:
        return "None"
    values = [str(int(v)) for v in solution.reshape(-1).tolist()]
    return ' '.join(values)


def write_solution_record(record_path, instance_name, is_minimize, best_round, best_sample, var_names):
    best_result = best_overall_result(best_round, best_sample, is_minimize)
    with open(record_path, 'w') as f:
        f.write(f"instance: {instance_name}\n")
        f.write(f"objective_sense: {'min' if is_minimize else 'max'}\n")
        f.write(f"round_best: {format_best_result(best_round)}\n")
        f.write(f"sample_best: {format_best_result(best_sample)}\n")

        if best_result is None:
            f.write("overall_best: None\n")
            return

        solution = best_result['solution'].reshape(-1)
        f.write(f"overall_best_method: {best_result['method']}\n")
        f.write(f"overall_best_obj: {best_result['obj']:.12g}\n")
        f.write(f"overall_best_epoch: {best_result['epoch']}\n")
        f.write(f"overall_best_time_seconds: {best_result['time']:.6f}\n")
        f.write(f"n_vars: {solution.numel()}\n")
        f.write(f"n_ones: {count_selected(solution)}\n")
        f.write("solution_vector_raw_order:\n")
        f.write(solution_vector_text(solution) + '\n')
        f.write("variable_values:\n")

        if var_names is not None and len(var_names) == solution.numel():
            for name, value in zip(var_names, solution.tolist()):
                f.write(f"{name} {int(value)}\n")
        else:
            for idx, value in enumerate(solution.tolist()):
                f.write(f"x[{idx}] {int(value)}\n")


# ============================================================
#  Main
# ============================================================

def main():
    parser = get_parser()
    args = parser.parse_args()

    device = args.device
    problem_type = args.problem_type
    batch_size = 1

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    save_name = (
        f'GS_lr{args.lr_output}_{args.lr_inner}'
        f'_bs{batch_size}_s{args.num_samples}'
        f'_mu{args.mu_init}_{args.mu_step_size}_{args.mu_min}_{args.mu_max}_{args.mu_value}'
        f'_emb{args.emb_size}_dep{args.depth}'
        f'_ICC{args.Intra_Constraint_Competitive}'
        f'_{args.loss_config}'
    )

    ins_dir = os.path.join(args.instance_dir, problem_type)
    if not os.path.isdir(ins_dir):
        ins_dir = args.instance_dir
    if not os.path.isdir(ins_dir):
        raise FileNotFoundError(f"Instance directory not found: {ins_dir}")

    all_instances = sorted(
        os.path.join(ins_dir, f)
        for f in os.listdir(ins_dir)
        if f.endswith(('.lp', '.mps'))
    )
    if not all_instances:
        raise FileNotFoundError(f"No .lp/.mps instances found in {ins_dir}")

    if args.resume_from is not None and len(all_instances) != 1:
        raise ValueError("--resume_from is only supported when --instance_dir contains one instance")

    solution_save_path = os.path.join(args.solution_save_dir, problem_type, save_name)
    os.makedirs(solution_save_path, exist_ok=True)

    model_save_path = None
    log_save_path = None
    summary_log = None
    temp_cache = None
    if args.record_artifacts:
        model_save_path = os.path.join(args.model_save_dir, problem_type)
        log_save_path = os.path.join(args.log_save_dir, problem_type)
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(log_save_path, exist_ok=True)
        summary_log = open(os.path.join(log_save_path, f'{save_name}_summary.log'), 'w')
    elif args.cache_dir is None:
        temp_cache = tempfile.TemporaryDirectory()

    try:
        for instance_idx, instance_path in enumerate(all_instances, start=1):
            instance_name = os.path.basename(instance_path)
            instance_stem = safe_instance_stem(instance_path)

            if args.cache_dir is not None:
                cache_dir = args.cache_dir
            elif args.record_artifacts:
                cache_dir = os.path.join(log_save_path, 'unsup_cache')
            else:
                cache_dir = temp_cache.name

            dataset = UnsupervisedGraphDataset([instance_path], cache_dir=cache_dir)
            instance_graph = dataset.get(0)
            instance_is_minimize = graph_is_minimize(instance_graph)
            raw_var_names = getattr(instance_graph, 'raw_var_names', None)

            data_loader = torch_geometric.loader.DataLoader(
                dataset, batch_size=1, shuffle=False,
                num_workers=args.num_workers,
                follow_batch=["constraint_features", "variable_features"],
            )

            model = build_model(args, device)
            optimizer = build_optimizer(model, args)
            scheduler = build_scheduler(optimizer, args)
            mu = args.mu_init
            start_epoch = 0

            if args.resume_from is not None:
                assert os.path.isfile(args.resume_from), f"Checkpoint not found: {args.resume_from}"
                ckpt = torch.load(args.resume_from, map_location=device)
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'])
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    if scheduler is not None and 'scheduler_state_dict' in ckpt:
                        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                    mu = ckpt.get('mu', mu)
                    start_epoch = ckpt.get('epoch', 0) + 1
                else:
                    model.load_state_dict(ckpt)

            tb_writer = None
            if args.record_artifacts and HAS_TENSORBOARD:
                tb_dir = os.path.join(args.tensorboard_dir, problem_type, save_name, instance_stem)
                os.makedirs(tb_dir, exist_ok=True)
                tb_writer = SummaryWriter(tb_dir)

            best_round = None
            best_sample = None
            cumulative_epoch_time = 0.0
            patience_counter = 0

            for epoch in range(start_epoch, args.num_epochs):
                t0 = time.time()
                epoch_loss, epoch_obj, epoch_cons, num_graphs = train_epoch(
                    model, data_loader, optimizer,
                    mu, args.num_samples, args.loss_config,
                    args.grad_clip_norm, device,
                )

                if scheduler is not None:
                    scheduler.step()

                avg_cons = epoch_cons / max(num_graphs, 1)
                mu = mu + args.mu_step_size * (avg_cons - args.mu_value)
                mu = max(min(mu, args.mu_max), args.mu_min)

                epoch_elapsed = time.time() - t0
                cumulative_epoch_time += epoch_elapsed

                eval_result = evaluate_current_instance(
                    model, data_loader, args.n_eval_samples, device
                )
                is_minimize = instance_is_minimize
                improved = False

                if eval_result['round_feasible']:
                    if (best_round is None or objective_is_better(
                            eval_result['round_obj'], best_round['obj'], is_minimize)):
                        best_round = {
                            'obj': eval_result['round_obj'],
                            'solution': eval_result['round_solution'],
                            'epoch': epoch,
                            'time': cumulative_epoch_time,
                        }
                        improved = True

                if eval_result['sample_n_feasible'] > 0:
                    if (best_sample is None or objective_is_better(
                            eval_result['sample_obj'], best_sample['obj'], is_minimize)):
                        best_sample = {
                            'obj': eval_result['sample_obj'],
                            'solution': eval_result['sample_solution'],
                            'epoch': epoch,
                            'time': cumulative_epoch_time,
                        }
                        improved = True

                if improved:
                    patience_counter = 0
                    if args.record_artifacts:
                        torch.save(
                            model.state_dict(),
                            os.path.join(model_save_path, f'{save_name}_{instance_stem}_model_best.pth')
                        )
                else:
                    patience_counter += 1

                if tb_writer is not None:
                    tb_writer.add_scalar('Loss/train_loss', epoch_loss / max(num_graphs, 1), epoch)
                    tb_writer.add_scalar('Loss/train_obj', epoch_obj / max(num_graphs, 1), epoch)
                    tb_writer.add_scalar('Loss/train_cons', avg_cons, epoch)
                    tb_writer.add_scalar('Params/mu', mu, epoch)
                    if best_round is not None:
                        tb_writer.add_scalar('Round/best_obj', best_round['obj'], epoch)
                    if best_sample is not None:
                        tb_writer.add_scalar('Sample/best_obj', best_sample['obj'], epoch)

                if args.record_artifacts:
                    full_ckpt = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'mu': mu,
                    }
                    if scheduler is not None:
                        full_ckpt['scheduler_state_dict'] = scheduler.state_dict()
                    torch.save(
                        full_ckpt,
                        os.path.join(model_save_path, f'{save_name}_{instance_stem}_model_last.pth')
                    )

                if patience_counter >= args.patience and epoch > args.patience:
                    break

            if tb_writer is not None:
                tb_writer.close()

            record_path = os.path.join(solution_save_path, f'{instance_stem}_solution.txt')
            write_solution_record(
                record_path, instance_name, instance_is_minimize,
                best_round, best_sample, raw_var_names,
            )

            summary = (
                f"Instance {instance_idx}/{len(all_instances)}: {instance_name}\n"
                f"  Round best: {format_best_result(best_round)}\n"
                f"  Sample best: {format_best_result(best_sample)}\n"
                f"  Solution record: {record_path}"
            )
            print(summary)
            write_line(summary_log, summary)

    finally:
        if summary_log is not None:
            summary_log.close()
        if temp_cache is not None:
            temp_cache.cleanup()


if __name__ == '__main__':
    main()
