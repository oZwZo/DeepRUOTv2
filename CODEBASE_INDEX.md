# DeepRUOTv2 Codebase Index

A comprehensive index of modules, scripts, functions, and classes for the DeepRUOT codebase.

---

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Core Module: DeepRUOT/](#core-module-deepruot)
   - [models.py](#modelspy)
   - [train.py](#trainpy)
   - [losses.py](#lossespy)
   - [utils.py](#utilspy)
   - [eval.py](#evalpy)
   - [exp.py](#exppy)
   - [constants.py](#constantspy)
3. [Training Scripts](#training-scripts)
4. [Configuration Files](#configuration-files)
5. [Data Files](#data-files)
6. [Evaluation & Analysis](#evaluation--analysis)

---

## Directory Structure

```
DeepRUOTv2/
├── DeepRUOT/                    # Core Python module
│   ├── __init__.py
│   ├── models.py               # Neural network architectures
│   ├── train.py                # Training functions
│   ├── losses.py               # Loss function implementations
│   ├── utils.py                # Utility functions & classes
│   ├── eval.py                 # Evaluation & trajectory generation
│   ├── exp.py                  # Experiment setup utilities
│   ├── constants.py            # Path constants
│   └── config/
│       └── default_config.py   # Default configuration
├── config/                      # Dataset-specific YAML configs
│   ├── weinreb_config.yaml
│   ├── veres_config.yaml
│   ├── eb_config.yaml
│   ├── emt_config.yaml
│   └── simulation_config.yaml
├── data/                        # Training datasets (CSV)
├── evaluation/                  # Jupyter notebooks for analysis
│   ├── analysis.ipynb
│   └── plot.ipynb
├── figures/                     # Generated figures
├── results/                     # Experiment outputs
├── train_RUOT.py               # Main training entry point
├── README.md
└── requirements.txt
```

---

## Core Module: DeepRUOT/

### models.py

**Purpose:** Neural network architectures for the DeepRUOT framework.

#### Classes

| Class | Line | Description | Input → Output |
|-------|------|-------------|----------------|
| `velocityNet` | 9-98 | Learns velocity field v(x,t) = dx/dt | (t, x) → v ∈ ℝᴰ |
| `growthNet` | 100-126 | Learns growth rate g(x,t) for cell birth/death | (t, x) → g ∈ ℝ |
| `scoreNet` | 128-154 | Legacy score network (outputs scalar) | (t, x) → s ∈ ℝ |
| `dediffusionNet` | 156-182 | State-dependent diffusion network | (t, x) → σ ∈ ℝ |
| `indediffusionNet` | 184-210 | Time-dependent diffusion σ(t) | t → σ ∈ ℝ |
| `FNet` | 212-232 | Main model combining all networks | (t, z) → (v, g, s, d) |
| `ODEFunc` | 256-263 | Simple ODE function (velocity only) | t, z → dz/dt |
| `ODEFunc2` | 236-254 | ODE function with mass tracking | t, (z, lnw, m) → (dz/dt, dlnw/dt, dm/dt) |
| `ODEFunc3` | 301-355 | ODE function with score network | t, (z, lnw, m) → (dz/dt, dlnw/dt, dm/dt) |
| `scoreNet2` | 265-297 | Score network with gradient computation | (t, x) → s ∈ ℝ, ∇s ∈ ℝᴰ |

#### Key Methods

| Method | Class | Line | Description |
|--------|-------|------|-------------|
| `forward(t, x)` | velocityNet | 62-98 | Compute velocity given time and position |
| `forward(t, x)` | growthNet | 121-126 | Compute growth rate |
| `compute_gradient(t, x)` | scoreNet2 | 291-297 | Compute ∇s = ∇logρ via autograd |

#### Architecture Details

```python
# velocityNet architecture
Input: [batch, in_out_dim + 1]  # x concatenated with t
Hidden: [hidden_dim] × n_hiddens layers with activation
Output: [batch, in_out_dim]     # velocity vector

# Optional: use_spatial=True splits into spatial + gene velocity
```

---

### train.py

**Purpose:** Training loop implementations for different phases.

#### Functions

| Function | Line | Purpose | Key Parameters |
|----------|------|---------|----------------|
| `train_un1` | 23-204 | Basic training loop (legacy) | model, df, groups, optimizer, n_batches |
| `train_un1_reduce` | 208-513 | Pretraining phase with reduced complexity | lambda_ot, lambda_mass, use_mass |
| `train_all` | 519-935 | Final joint training with all components | sf2m_score_model, use_pinn, sigmaa |

#### train_un1_reduce Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | nn.Module | required | FNet model |
| `df` | DataFrame | required | Data with 'samples' column |
| `groups` | list | required | Time point indices |
| `optimizer` | Optimizer | required | PyTorch optimizer |
| `n_batches` | int | 20 | Number of training batches |
| `criterion` | Loss | MMD_loss | Loss function |
| `sample_size` | tuple | (100,) | Samples per batch |
| `lambda_ot` | float | 0.1 | OT loss weight |
| `lambda_mass` | float | 1 | Mass loss weight |
| `lambda_energy` | float | 0.01 | Energy loss weight |
| `use_mass` | bool | True | Enable growth modeling |
| `hold_one_out` | bool | False | Leave-one-out validation |
| `best_model_path` | str | None | Path to save best model |

#### train_all Additional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sf2m_score_model` | nn.Module | None | Score network |
| `use_pinn` | bool | False | Enable PINN loss |
| `lambda_pinn` | float | 1 | PINN loss weight |
| `sigmaa` | float | 0.1 | Noise strength |
| `scheduler` | Scheduler | None | Learning rate scheduler |

#### Training Flow

```
train_un1_reduce (Pretraining):
  For each batch:
    For each (t0, t1) step:
      1. Sample data at t0, t1
      2. Solve ODE: x_t = odeint(ODEFunc2, (x0, lnw0, m0), [t0, t1])
      3. Compute OT loss: Wasserstein(x_t[-1], data_t1)
      4. Compute mass loss: cal_mass_loss_reduce(...)
      5. Backward + optimizer step

train_all (Final Training):
  For each batch:
    For each (t0, t1) step:
      1. Sample data, initialize weights
      2. Solve ODE: x_t = odeint(ODEFunc3, ..., [t0, t1])
      3. Compute OT loss via Sinkhorn
      4. Compute mass losses (global + local)
      5. If use_pinn: Compute PINN loss (Fokker-Planck)
      6. Combined loss backward + step
```

---

### losses.py

**Purpose:** Loss function implementations for training.

#### Classes

| Class | Line | Description | Usage |
|-------|------|-------------|-------|
| `MMD_loss` | 9-43 | Maximum Mean Discrepancy with Gaussian kernel | Distribution matching |
| `OT_loss1` | 51-97 | Optimal Transport loss (EMD/Sinkhorn) | Trajectory matching |
| `Density_loss` | 101-124 | k-NN based density regularization | Local structure |
| `Local_density_loss` | 127-143 | Local density variant | Alternative density loss |

#### MMD_loss Methods

| Method | Line | Description |
|--------|------|-------------|
| `guassian_kernel(source, target, ...)` | 20-33 | Compute multi-scale Gaussian kernel |
| `forward(source, target)` | 35-43 | Compute MMD distance |

#### OT_loss1 Methods

| Method | Line | Description |
|--------|------|-------------|
| `__call__(source, target, mu, nu, sigma)` | 60-97 | Compute OT loss with given marginals |

#### Supported OT Methods

```python
_valid = ['emd', 'sinkhorn', 'sinkhorn_knopp_unbalanced']
```

---

### utils.py

**Purpose:** Utility functions, data handling, flow matching, and SDE integration.

#### Data Handling Functions

| Function | Line | Description |
|----------|------|-------------|
| `load_and_merge_config(config_path)` | 15-53 | Load YAML config with defaults |
| `group_extract(df, group, index, groupby)` | 58-59 | Extract data for specific time group |
| `sample(data, group, size, replace, to_torch, device)` | 61-69 | Sample points from group |
| `to_np(data)` | 71-72 | Convert tensor to numpy |
| `generate_steps(groups)` | 74-75 | Generate (t0, t1) pairs from groups |
| `set_seeds(seed)` | 77-80 | Set random seeds for reproducibility |
| `config_hold_out(df, hold_out, hold_one_out)` | 82-96 | Configure hold-out validation |
| `config_criterion(criterion_name, use_cuda)` | 98-110 | Create loss function by name |

#### Mass Loss Functions

| Function | Line | Description |
|----------|------|-------------|
| `cal_mass_loss(data_t1, x_t_last, lnw_t_last, relative_mass, batch_size)` | 261-282 | Original local mass matching |
| `cal_mass_loss_reduce(data_t1, x_t_last, lnw_t_last, relative_mass, batch_size, dim_reducer)` | 287-331 | Mass loss with optional dimensionality reduction |

#### Flow Matching Classes

| Class | Line | Description |
|-------|------|-------------|
| `ConditionalFlowMatcher` | 689-738 | Base CFM class |
| `ExactOptimalTransportConditionalFlowMatcher` | 741-764 | CFM with exact OT coupling |
| `TargetConditionalFlowMatcher` | 767-784 | Target-based CFM |
| `SchrodingerBridgeConditionalFlowMatcher` | 787-826 | CFM with Schrödinger Bridge |
| `VariancePreservingConditionalFlowMatcher` | 829-840 | Variance-preserving CFM |

#### OTPlanSampler Class

| Method | Line | Description |
|--------|------|-------------|
| `__init__(method, reg, reg_m, ...)` | 544-568 | Initialize with OT method |
| `get_map(x0, x1)` | 570-591 | Compute OT plan matrix |
| `sample_map(pi, batch_size, replace)` | 593-600 | Sample from OT plan |
| `sample_plan(x0, x1, replace)` | 602-606 | Sample paired points via OT |
| `sample_plan_with_labels(...)` | 608-617 | Sample with labels |

#### Trajectory Generation Functions

| Function | Line | Description |
|----------|------|-------------|
| `generate_state_trajectory(X, n_times, batch_size, f_net, time, device)` | 845-855 | Generate reference trajectory |
| `get_batch(FM, X, trajectory, batch_size, n_times, ...)` | 858-887 | Create training batch (legacy) |
| `get_batch_size(FM, X, trajectory, batch_size, time, ...)` | 912-955 | Create training batch with time scaling |

#### SDE Integration Functions

| Function | Line | Description |
|----------|------|-------------|
| `euler_sdeint(sde, initial_state, dt, ts)` | 958-998 | Euler-Maruyama SDE integration |
| `euler_sdeint_split(sde, initial_state, dt, ts, noise_std)` | 1001-1077 | SDE with cell splitting/extinction |

#### Helper Functions

| Function | Line | Description |
|----------|------|-------------|
| `trace_df_dz(f, z)` | 521-526 | Compute trace of Jacobian df/dz |
| `density1(x, datatime0, device)` | 891-910 | Multimodal Gaussian density |
| `wasserstein(x0, x1, method, reg, power)` | 640-670 | Compute Wasserstein distance |

---

### eval.py

**Purpose:** Evaluation metrics and trajectory generation.

#### Functions

| Function | Description |
|----------|-------------|
| `generate_trajectories_sde(...)` | Generate trajectories using SDE integration |
| Evaluation metrics computation | W1 distance, TMV (Total Movement Variance) |

---

### exp.py

**Purpose:** Experiment setup and logging utilities.

#### Functions

| Function | Description |
|----------|-------------|
| `setup_exp(output_dir, config, name)` | Create experiment directory and logger |

---

### constants.py

**Purpose:** Define project path constants.

```python
ROOT_DIR    # Project root directory
DATA_DIR    # Data directory
NTBK_DIR    # Notebooks directory
IMGS_DIR    # Images directory
RES_DIR     # Results directory
```

---

## Training Scripts

### train_RUOT.py

**Purpose:** Main training entry point with complete pipeline.

#### Classes

| Class | Line | Description |
|-------|------|-------------|
| `TrainingPipeline` | 32-394 | Complete training pipeline manager |

#### TrainingPipeline Methods

| Method | Line | Description |
|--------|------|-------------|
| `__init__(config)` | 33-53 | Initialize pipeline with config |
| `_setup_experiment()` | 55-62 | Create experiment directory |
| `_setup_models()` | 64-79 | Initialize FNet and scoreNet2 |
| `_load_data()` | 81-84 | Load CSV data |
| `_setup_training()` | 86-104 | Setup training parameters |
| `pretrain()` | 106-176 | Phase 1: Pretrain velocity & growth |
| `train_score_model()` | 178-257 | Phase 2: Train score network |
| `final_training()` | 259-334 | Phase 3: Joint training |
| `clean_up()` | 336-344 | Remove intermediate files |
| `train()` | 346-360 | Run complete training pipeline |
| `evaluate()` | 362-394 | Generate and evaluate trajectories |

#### Main Function

```python
def main():
    # Usage: python train_RUOT.py --config config/veres_config.yaml
    config = load_and_merge_config(args.config)
    pipeline = TrainingPipeline(config)
    pipeline.train()
    pipeline.evaluate()
```

#### Training Pipeline Flow

```
TrainingPipeline.train():
  1. pretrain()           → Phase 1: v_net, g_net initialization
  2. train_score_model()  → Phase 2: score network via CFM
  3. final_training()     → Phase 3: Joint optimization
  4. clean_up()           → Remove checkpoints
  5. evaluate()           → Generate & evaluate trajectories
```

---

## Configuration Files

### config/*.yaml

#### Common Structure

```yaml
# Runtime parameters
use_pinn: false              # Enable PINN refinement
sample_with_replacement: false
device: 'cuda'
sample_size: 1024
use_mass: true               # Enable growth/death

# Experiment settings
exp:
  name: 'experiment_name'
  output_dir: 'results'

# Data settings
data:
  file_path: 'filename.csv'
  dim: 30                    # Number of features
  hold_one_out: false
  hold_out: 1

# Model architecture
model:
  in_out_dim: 30
  hidden_dim: 400
  n_hiddens: 2
  activation: 'leakyrelu'
  score_hidden_dim: 128

# Pretraining phase
pretrain:
  epochs: 500
  lr: 0.0001
  lambda_ot: 1.0
  lambda_mass: 0.01
  lambda_energy: 0.0

# Score training phase
score_train:
  epochs: 3001
  lr: 0.0001
  lambda_penalty: 1
  sigma: 0.1
  score_batch_size: 512

# Final training phase
train:
  epochs: 500
  lr: 0.0001
  lambda_ot: 10
  lambda_mass: 10
  lambda_energy: 0.01
  lambda_pinn: 100
  lambda_initial: 0.1
  scheduler_step_size: 100
  scheduler_gamma: 0.8
```

### Available Configs

| Config | Dataset | Dimensions | Description |
|--------|---------|------------|-------------|
| `weinreb_config.yaml` | Mouse Blood Hematopoiesis | 50D | Blood cell differentiation |
| `veres_config.yaml` | Pancreatic β-cell | 30D | Pancreas development |
| `eb_config.yaml` | Embryoid Body | 50D | EB differentiation |
| `emt_config.yaml` | A549 EMT | 10D | Epithelial-mesenchymal transition |
| `simulation_config.yaml` | Simulated data | - | Validation dataset |

---

## Data Files

### data/*.csv

#### Required Format

```csv
samples,x1,x2,x3,...,xD
0,0.123,0.456,0.789,...
0,0.234,0.567,0.890,...
1,0.345,0.678,0.901,...
1,0.456,0.789,0.012,...
2,...
```

| Column | Type | Description |
|--------|------|-------------|
| `samples` | int | Time point index (0, 1, 2, ...) |
| `x1`...`xD` | float | Gene expression / PCA features |

### Available Datasets

| File | Cells | Timepoints | Dimensions |
|------|-------|------------|------------|
| `Weinreb_alltime.csv` | ~5000 | 7 | 50 |
| `Veres_alltime.csv` | ~3000 | 5 | 30 |
| `eb_noscale.csv` | ~10000 | 5 | 50 |
| `emt.csv` | ~2000 | 8 | 10 |
| `simulation_gene.csv` | varies | varies | varies |

---

## Evaluation & Analysis

### evaluation/analysis.ipynb

- Trajectory visualization
- Gene expression trend analysis
- Cell fate probability inference
- Benchmark comparisons

### evaluation/plot.ipynb

- Figure generation for papers
- Loss curve visualization
- Comparative plots

---

## Key Equations Reference

### Neural Network Outputs

| Network | Output | Mathematical Meaning |
|---------|--------|---------------------|
| v_net | v(x,t) | Velocity field: dx/dt |
| g_net | g(x,t) | Growth rate: d(ln w)/dt |
| s_net | s(x,t) | Score: log ρ(x,t) |
| ∇s_net | ∇s(x,t) | Score gradient: ∇log ρ |

### ODE Dynamics

```
dz/dt = v(z,t)                           # Position dynamics
d(ln w)/dt = g(z,t)                      # Weight (mass) dynamics
dm/dt = (||v||²/2 + ||∇s||²/2 + g²) * w  # Energy accumulation
```

### Loss Components

```python
# Total loss
L = lambda_ot * L_ot + lambda_mass * L_mass + lambda_energy * L_energy + lambda_pinn * L_pinn

# OT loss (Sinkhorn)
L_ot = Sinkhorn(mu, x_pred, nu, x_true)

# Mass loss
L_mass = L_global_mass + L_local_mass
L_global_mass = ||sum(exp(lnw)) - relative_mass||²
L_local_mass = ||weights - relative_mass * relative_count||²

# PINN loss (Fokker-Planck)
L_pinn = |ρ_t + ∇·(vρ) - gρ|
```

---

## Quick Reference

### Running Training

```bash
python train_RUOT.py --config config/veres_config.yaml
```

### Key Files to Modify

| Task | File |
|------|------|
| Change model architecture | `DeepRUOT/models.py` |
| Modify training loop | `DeepRUOT/train.py` |
| Add new loss function | `DeepRUOT/losses.py` |
| Change data loading | `DeepRUOT/utils.py` |
| Adjust hyperparameters | `config/*.yaml` |
| New dataset | `data/*.csv` + new config |

### Common Parameters to Tune

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| `sigma` | Stochasticity | 0.0 - 0.5 |
| `hidden_dim` | Model capacity | 128 - 512 |
| `lambda_ot` | Trajectory matching | 1 - 20 |
| `lambda_mass` | Growth constraint | 0.01 - 10 |
| `use_pinn` | Physics constraint | true/false |
| `sample_size` | Number of particles | 512 - 2048 |
