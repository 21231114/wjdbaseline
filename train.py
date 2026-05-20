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
        best_feasible_obj: best objective among feasible solutions (inf if none).
        best_obj: best objective among all solutions.
        mean_obj: mean objective over all solutions.
        mean_feasible_obj: mean objective among feasible solutions (inf if none).
        n_feasible: number of feasible solutions found.
    """
    n_raw_vars = graph.obj_coeffs.shape[0]
    n_cons = graph.raw_n_cons if isinstance(graph.raw_n_cons, int) else graph.raw_n_cons.item()

    logits_raw = map_logits_to_raw(logits_gnn, graph.gnn_to_raw_map, n_raw_vars, device)

    A = build_dense_A(graph.raw_cons_indices, graph.raw_cons_values, n_cons, n_raw_vars, device)
    b = graph.raw_rhs.to(device).reshape(-1, 1)
    c = graph.obj_coeffs.to(device).reshape(-1, 1)

    # Sample solutions
    xx = gumbel_sample(logits_raw, n_eval_samples, tau=1.0).float().reshape(n_eval_samples, -1)

    # Objectives for all samples
    objs = (xx @ c).squeeze(-1)  # [n_eval_samples]

    # Find feasible solutions: all constraints satisfied
    violations = torch.relu(A @ xx.T - b).sum(dim=0)  # [n_eval_samples]
    feasible_mask = (violations == 0)
    n_feasible = feasible_mask.sum().item()

    if n_feasible > 0:
        best_feasible_obj = objs[feasible_mask].min().item()
        mean_feasible_obj = objs[feasible_mask].mean().item()
    else:
        best_feasible_obj = float('inf')
        mean_feasible_obj = float('inf')

    best_obj = objs.min().item()
    mean_obj = objs.mean().item()

    return best_feasible_obj, best_obj, mean_obj, mean_feasible_obj, n_feasible


@torch.no_grad()
def evaluate_by_rounding(logits_gnn, graph, device):
    """
    Evaluate a graph by rounding sigmoid(logits) to 0/1 and checking feasibility.

    Returns:
        is_feasible: whether the rounded solution satisfies all constraints.
        obj_val: objective value of the rounded solution.
    """
    n_raw_vars = graph.obj_coeffs.shape[0]
    n_cons = graph.raw_n_cons if isinstance(graph.raw_n_cons, int) else graph.raw_n_cons.item()

    logits_raw = map_logits_to_raw(logits_gnn, graph.gnn_to_raw_map, n_raw_vars, device)

    A = build_dense_A(graph.raw_cons_indices, graph.raw_cons_values, n_cons, n_raw_vars, device)
    b = graph.raw_rhs.to(device).reshape(-1, 1)
    c = graph.obj_coeffs.to(device).reshape(-1, 1)

    # Round sigmoid(logits) to 0/1
    x_round = torch.round(torch.sigmoid(logits_raw)).reshape(1, -1)

    # Objective
    obj_val = (x_round @ c).item()

    # Check feasibility: all constraints Ax <= b
    violation = torch.relu(A @ x_round.T - b).sum().item()
    is_feasible = (violation == 0)

    return is_feasible, obj_val


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
            best_feas, best_obj, mean_obj, mean_feas_obj, n_feas = evaluate_by_sampling(
                logits_per_graph[i], g, n_eval_samples, device
            )
            is_round_feasible, round_obj = evaluate_by_rounding(
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
    parser.add_argument("--mu_max", type=float, default=0.8,
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
    parser.add_argument("--tensorboard_dir", default="./tb_logs",
                        help="TensorBoard log directory")

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
#  Main
# ============================================================

def main():
    parser = get_parser()
    args = parser.parse_args()

    device = args.device
    problem_type = args.problem_type
    batch_size = args.batch_size or TASK_BATCH_SIZE.get(problem_type, 1)

    # Reproducibility
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

    # Directories
    model_save_path = os.path.join(args.model_save_dir, problem_type)
    log_save_path = os.path.join(args.log_save_dir, problem_type)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)

    log_file = open(f'{log_save_path}/{save_name}_train.log', 'w')

    # TensorBoard
    tb_writer = None
    if HAS_TENSORBOARD:
        tb_dir = os.path.join(args.tensorboard_dir, problem_type, save_name)
        os.makedirs(tb_dir, exist_ok=True)
        tb_writer = SummaryWriter(tb_dir)
        print(f"TensorBoard logging to {tb_dir}")

    # ---- Data loading ----
    # Try instance_dir/problem_type first; if it doesn't exist, use instance_dir directly
    ins_dir = os.path.join(args.instance_dir, problem_type)
    if not os.path.isdir(ins_dir):
        ins_dir = args.instance_dir
    all_instances = sorted([
        os.path.join(ins_dir, f)
        for f in os.listdir(ins_dir)
        if f.endswith(('.lp', '.mps'))
    ])

    random.shuffle(all_instances)
    split = int(0.8 * len(all_instances))
    # Single-instance case: use the same instance for both train and valid
    if split == 0 or len(all_instances) == 1:
        train_files = all_instances
        valid_files = all_instances
    else:
        train_files = all_instances[:split]
        valid_files = all_instances[split:]

    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = os.path.join(args.log_save_dir, problem_type, 'unsup_cache')

    train_data = UnsupervisedGraphDataset(train_files, cache_dir=cache_dir)
    valid_data = UnsupervisedGraphDataset(valid_files, cache_dir=cache_dir)

    train_loader = torch_geometric.loader.DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers,
        follow_batch=["constraint_features", "variable_features"],
    )
    valid_loader = torch_geometric.loader.DataLoader(
        valid_data, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers,
        follow_batch=["constraint_features", "variable_features"],
    )

    print(f"Train instances: {len(train_files)}, Valid instances: {len(valid_files)}")
    print(f"Batch size: {batch_size}")

    # Objective sense
    first_data = train_data[0]
    if hasattr(first_data, 'obj_sense_min'):
        is_minimize = bool(first_data.obj_sense_min.item())
    else:
        _raw = extract_raw_ilp(train_files[0])
        is_minimize = _raw['obj_sense_min']
    obj_sign = 1.0 if is_minimize else -1.0
    opt_dir = "MINIMIZE" if is_minimize else "MAXIMIZE"
    print(f"Objective sense: {opt_dir}")

    # ---- Model ----
    model = GNNPolicy(
        emb_size=args.emb_size,
        cons_nfeats=args.cons_nfeats,
        edge_nfeats=args.edge_nfeats,
        var_nfeats=args.var_nfeats,
        depth=args.depth,
        Intra_Constraint_Competitive=args.Intra_Constraint_Competitive,
    ).to(device)

    # ---- Optimizer with separate learning rates ----
    output_param_ids = set()
    for layer in (model.vars_output_layer, model.cons_output_layer):
        for p in layer.parameters():
            output_param_ids.add(id(p))
    other_params = [p for p in model.parameters() if id(p) not in output_param_ids]

    optimizer = optim.Adam([
        {'params': model.vars_output_layer.parameters(), 'lr': args.lr_output},
        {'params': model.cons_output_layer.parameters(), 'lr': args.lr_output},
        {'params': other_params, 'lr': args.lr_inner},
    ], weight_decay=args.weight_decay)

    # ---- LR Schedule ----
    if args.lr_schedule == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.cos_T, eta_min=args.cos_min
        )
    elif args.lr_schedule == 'cosrestart':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.cos_T, T_mult=2, eta_min=args.cos_min
        )
    elif args.lr_schedule == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_anneal_factor
        )
    else:
        scheduler = None

    # ---- Dynamic penalty state ----
    mu = args.mu_init
    start_epoch = 0

    # ---- Resume from checkpoint ----
    if args.resume_from is not None:
        assert os.path.isfile(args.resume_from), f"Checkpoint not found: {args.resume_from}"
        print(f"Loading checkpoint from {args.resume_from} ...")
        ckpt = torch.load(args.resume_from, map_location=device)

        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if scheduler is not None and 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            mu = ckpt.get('mu', mu)
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f"  Resumed: epoch={start_epoch}, mu={mu:.4f}")
        else:
            model.load_state_dict(ckpt)
            print("  Loaded model weights only. Training from epoch 0.")

    # ---- Training ----
    best_round_feasible = -1  # best number of rounding-feasible instances (maximize)
    best_round_obj = float('inf') if is_minimize else float('-inf')  # tiebreaker
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"Starting Gumbel-Softmax Training for {problem_type} ({opt_dir})")
    print(f"  lr_output={args.lr_output}, lr_inner={args.lr_inner}")
    print(f"  num_samples={args.num_samples}, loss_config={args.loss_config}")
    print(f"  mu_init={args.mu_init}, mu_step_size={args.mu_step_size}")
    print(f"  mu_range=[{args.mu_min}, {args.mu_max}], mu_value={args.mu_value}")
    print(f"  lr_schedule={args.lr_schedule}, cos_T={args.cos_T}")
    print(f"  batch_size={batch_size}, num_epochs={args.num_epochs}")
    print(f"{'='*70}\n")

    global_step = 0

    for epoch in range(start_epoch, args.num_epochs):
        t0 = time.time()

        # ---- Train ----
        epoch_loss, epoch_obj, epoch_cons, num_graphs = train_epoch(
            model, train_loader, optimizer,
            mu, args.num_samples, args.loss_config,
            args.grad_clip_norm, device,
        )

        # Update LR schedule (per epoch)
        if scheduler is not None:
            scheduler.step()

        # Update mu dynamically
        avg_cons = epoch_cons / max(num_graphs, 1)
        mu = mu + args.mu_step_size * (avg_cons - args.mu_value)
        mu = max(min(mu, args.mu_max), args.mu_min)

        avg_loss = epoch_loss / max(num_graphs, 1)
        avg_obj = epoch_obj / max(num_graphs, 1)

        # Get current learning rates
        lr_list = scheduler.get_last_lr() if scheduler else [args.lr_output, args.lr_output, args.lr_inner]

        elapsed = time.time() - t0

        # Log training
        train_log = (
            f"@epoch{epoch}  TIME:{elapsed:.1f}s\n"
            f"  [Train] Loss={avg_loss:.4f}  Obj={avg_obj:.4f}  "
            f"Cons={avg_cons:.4f}  mu={mu:.4f}  "
            f"lr_o={lr_list[0]:.2e}  lr_i={lr_list[-1]:.2e}"
        )

        # TensorBoard training
        if tb_writer is not None:
            tb_writer.add_scalar('Loss/train_loss', avg_loss, epoch)
            tb_writer.add_scalar('Loss/train_obj', avg_obj, epoch)
            tb_writer.add_scalar('Loss/train_cons', avg_cons, epoch)
            tb_writer.add_scalar('Params/mu', mu, epoch)
            tb_writer.add_scalar('Params/lr_o', lr_list[0], epoch)
            tb_writer.add_scalar('Params/lr_i', lr_list[-1], epoch)

        # ---- Validate ----
        val_metrics = None
        if (epoch + 1) % args.val_every == 0 or epoch == 0:
            val_metrics = valid_epoch(
                model, valid_loader, mu, args.num_samples,
                args.loss_config, args.n_eval_samples, device,
            )

            feas_rate = val_metrics['total_feasible'] / max(val_metrics['n_eval_samples'], 1)
            train_log += (
                f"\n  [Valid] Loss={val_metrics['loss']:.4f}  "
                f"Obj={val_metrics['obj']:.4f}  Cons={val_metrics['cons']:.4f}"
                f"\n  [Round] Feasible={val_metrics['round_n_feasible']}/{val_metrics['n_valid_instances']}  "
                f"AvgObj={val_metrics['round_avg_obj']:.4f}"
                f"\n  [Sample] BestFeasObj={val_metrics['best_feasible']:.4f}  "
                f"BestObj={val_metrics['best_obj']:.4f}  "
                f"MeanObj={val_metrics['mean_obj']:.4f}  "
                f"MeanFeasObj={val_metrics['mean_feasible_obj']:.4f}  "
                f"AvgBestFeasAll={val_metrics['avg_best_feas_all']:.4f}"
                f"\n          FeasInst={val_metrics['n_sample_feasible_instances']}/{val_metrics['n_valid_instances']}  "
                f"FeasSamples={val_metrics['total_feasible']}/{val_metrics['n_eval_samples']}  "
                f"FeasRate={feas_rate:.4f}"
            )

            # TensorBoard validation
            if tb_writer is not None:
                tb_writer.add_scalar('Loss/valid_loss', val_metrics['loss'], epoch)
                tb_writer.add_scalar('Loss/valid_obj', val_metrics['obj'], epoch)
                tb_writer.add_scalar('Loss/valid_cons', val_metrics['cons'], epoch)
                tb_writer.add_scalar('Round/feasible_count', val_metrics['round_n_feasible'], epoch)
                tb_writer.add_scalar('Round/avg_obj', val_metrics['round_avg_obj'], epoch)
                tb_writer.add_scalar('Sample/best_feasible_obj', val_metrics['best_feasible'], epoch)
                tb_writer.add_scalar('Sample/best_obj', val_metrics['best_obj'], epoch)
                tb_writer.add_scalar('Sample/mean_obj', val_metrics['mean_obj'], epoch)
                tb_writer.add_scalar('Sample/mean_feasible_obj', val_metrics['mean_feasible_obj'], epoch)
                tb_writer.add_scalar('Sample/feasible_instances', val_metrics['n_sample_feasible_instances'], epoch)
                tb_writer.add_scalar('Sample/feasible_rate', feas_rate, epoch)

            # Save best model: maximize rounding-feasible count, then best avg obj
            curr_round_feas = val_metrics['round_n_feasible']
            curr_round_obj = val_metrics['round_avg_obj']

            improved = False
            if curr_round_feas > best_round_feasible:
                improved = True
            elif curr_round_feas == best_round_feasible and curr_round_feas > 0:
                if is_minimize and curr_round_obj < best_round_obj:
                    improved = True
                elif not is_minimize and curr_round_obj > best_round_obj:
                    improved = True

            if improved:
                best_round_feasible = curr_round_feas
                best_round_obj = curr_round_obj
                patience_counter = 0
                torch.save(
                    model.state_dict(),
                    os.path.join(model_save_path, f'{save_name}_model_best.pth')
                )
                train_log += (f"\n  >> Best model saved (round_feasible="
                              f"{curr_round_feas}/{val_metrics['n_valid_instances']}, "
                              f"round_avg_obj={curr_round_obj:.4f})")
            else:
                patience_counter += 1

        # Save latest checkpoint (full, resumable)
        full_ckpt = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'mu': mu,
        }
        if scheduler is not None:
            full_ckpt['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(full_ckpt, os.path.join(model_save_path, f'{save_name}_model_last.pth'))

        print(train_log)
        log_file.write(train_log + '\n')
        log_file.flush()

        # Early stopping
        if (val_metrics is not None
                and patience_counter >= args.patience
                and epoch > args.patience):
            print(f"\nEarly stopping at epoch {epoch}: "
                  f"no improvement for {args.patience} epochs.")
            print(f"  Best round_feasible: {best_round_feasible}, "
                  f"Best round_avg_obj: {best_round_obj:.4f}")
            break

    log_file.close()
    if tb_writer is not None:
        tb_writer.close()
    print("Training completed.")


if __name__ == '__main__':
    main()
