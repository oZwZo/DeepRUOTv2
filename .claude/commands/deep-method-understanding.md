# Deep Method Understanding Skill

A comprehensive interactive workflow for thoroughly understanding a computational method, benchmarking it on your data, diagnosing issues, and suggesting improvements.

## When to Use

Invoke this skill when you need to:
- Understand a new deep learning/ML method from manuscript + code
- Benchmark the method on your own dataset
- Diagnose accuracy issues, theory-practice gaps, or data sensitivity problems
- Get theoretical and code-level improvement suggestions

## Prerequisites

Before starting, ensure you have:
- [ ] Manuscript/Paper available (PDF preferred)
- [ ] GitHub repository cloned locally
- [ ] NotebookLM notebook created with manuscript (expand to supplements later)
- [ ] Your own dataset accessible

---

## Phase 1: Context Foundation

**Goal**: Build comprehensive understanding of the method's theoretical foundations.

### Step 1.1: Core Concept Extraction
Use NotebookLM to answer:
- What is the core mathematical formulation?
- What problem does this method solve?
- What are the key assumptions and constraints?
- How does it differ from prior approaches?

### Step 1.2: Mathematical Framework Mapping
Extract and document:
- Input/Output specifications (shapes, types, constraints)
- Loss function(s) and their derivation
- Optimization approach (optimizer, learning rate schedule, etc.)
- Key hyperparameters and their theoretical significance
- Architectural innovations (new layers, attention mechanisms, etc.)

### Step 1.3: Literature Context
- What are the main prior methods this builds upon?
- What are the claimed advantages over baselines?
- What datasets were used in the original evaluation?

**Checkpoint**: Summarize understanding in 2-3 paragraphs. Ask clarifying questions if concepts are unclear.

---

## Phase 2: Code Architecture Analysis

**Goal**: Map the theoretical framework to actual implementation.

### Step 2.1: Repository Structure Mapping
```
# Analyze directory structure
Identify:
├── Model definitions (usually in models/, nets/, or network/)
├── Data loading/pipeline (data/, dataset/, dataloader/)
├── Training scripts (train.py, main.py, run/)
├── Evaluation/benchmarking (eval/, test/, benchmark/)
├── Utilities (utils/, lib/, helpers/)
└── Configuration (config/, configs/, args/)
```

### Step 2.2: Model Architecture Tracing
For each major component:
- Trace from `forward()` method through all layers
- Document tensor shapes at each stage
- Map code modules to paper sections/equations
- Identify any implementation deviations from paper

### Step 2.3: Data Pipeline Understanding
Document:
- Expected input format and preprocessing
- Data augmentation/normalization used
- Batch construction logic
- Any domain-specific transformations (e.g., for scRNA-seq: normalization, log-transform, HVG selection)

### Step 2.4: Training Loop Analysis
Extract:
- Loss computation details
- Gradient flow (any special handling?)
- Learning rate schedule
- Regularization techniques
- Early stopping / checkpoint logic

**Checkpoint**: Create a code-to-paper mapping table. Flag any discrepancies.

---

## Phase 3: Theory-Code Gap Analysis

**Goal**: Identify inconsistencies between theoretical claims and implementation.

### Step 3.1: Systematic Comparison
| Paper Claim | Code Implementation | Match? | Notes |
|-------------|---------------------|--------|-------|
| Loss = X + Y | Actual loss calculation | | |
| Layer dimensions | Actual layer sizes | | |
| Hyperparameters in Table X | Default config values | | |

### Step 3.2: Implicit Assumptions
Identify assumptions made in code but not explicitly stated:
- Hardware requirements (GPU memory, multi-GPU)
- Data characteristics assumed (size ranges, distribution)
- Numerical precision/stability considerations

### Step 3.3: Implementation Simplifications
- Any approximations to theoretical formulations?
- Any missing components from paper?
- Any additional components not in paper?

**Checkpoint**: Discuss findings. These gaps often explain theory-practice discrepancies.

---

## Phase 4: Benchmarking Setup

**Goal**: Establish reproducible evaluation on your dataset.

### Step 4.1: Data Preparation Protocol
For high-dimensional time series (e.g., scRNA-seq multi-timepoint):
1. Document your data format and characteristics
2. Map your data to expected input format
3. Implement any required preprocessing
4. Create train/val/test splits appropriate for your task

### Step 4.2: Baseline Selection
Identify appropriate baselines:
- Classical methods (PCA, simple neural networks)
- Domain-specific baselines (for scRNA-seq: Seurat, Scanpy methods)
- Prior state-of-the-art methods

### Step 4.3: Metric Selection
Define evaluation metrics aligned with your goals:
- Primary: Accuracy-related metrics for your task
- Secondary: Computational efficiency, robustness measures
- Domain-specific: Biological plausibility metrics for scRNA-seq

### Step 4.4: Configuration Template
```yaml
# benchmark_config.yaml template
dataset:
  path: /path/to/your/data
  format: [describe format]
  preprocessing: [list steps]

model:
  checkpoint: [pretrained or None]
  hyperparameters: [from paper defaults]

training:
  batch_size: [appropriate for your GPU]
  epochs: [max epochs]
  learning_rate: [from paper]
  early_stopping: [criteria]

evaluation:
  metrics: [list metrics]
  baseline_comparison: [list baselines]
```

**Checkpoint**: Review benchmarking plan before execution.

---

## Phase 5: Diagnostic Analysis

**Goal**: Systematically identify root causes of performance issues.

### Step 5.1: Issue Classification Framework

**Accuracy Issues Diagnosis:**
```
├── Data-related
│   ├── Distribution shift (train vs test)
│   ├── Insufficient data quantity
│   ├── Poor data quality (noise, missing values)
│   └── Inappropriate preprocessing
├── Model-related
│   ├── Underfitting (model too simple)
│   ├── Overfitting (model too complex)
│   ├── Architectural mismatch for data type
│   └── Initialization problems
└── Training-related
    ├── Learning rate too high/low
    ├── Insufficient training duration
    ├── Improper regularization
    └── Batch size issues
```

**Theory-Practice Gap Diagnosis:**
```
├── Assumption violations
│   ├── Data doesn't meet theoretical assumptions
│   ├── Scale/dimension mismatch
│   └── Distribution assumptions violated
├── Implementation issues
│   ├── Bugs in core algorithm
│   ├── Numerical instability
│   └── Missing components
└── Evaluation issues
    ├── Metrics don't measure what paper claims
    ├── Inappropriate baselines
    └── Data leakage in evaluation
```

**Data Sensitivity Diagnosis:**
```
├── Domain-specific (scRNA-seq)
│   ├── Batch effects not handled
│   ├── Cell type imbalance
│   ├── Dropout/zero-inflation issues
│   └── Temporal misalignment
└── General
    ├── Feature scale sensitivity
    ├── Sample size sensitivity
    └── Hyperparameter sensitivity
```

### Step 5.2: Diagnostic Experiments
Design targeted experiments to isolate issues:

```python
# Diagnostic experiment template
def diagnostic_experiment(issue_type, hypothesis):
    """
    1. State hypothesis clearly
    2. Design minimal experiment to test
    3. Run experiment with controlled variables
    4. Analyze results
    5. Update understanding
    """
    pass
```

Example diagnostics:
- **Learning curve analysis**: Plot train/val loss to detect over/underfitting
- **Ablation studies**: Remove components to identify critical parts
- **Data subset experiments**: Test on varying data sizes/distributions
- **Gradient flow analysis**: Check for vanishing/exploding gradients
- **Attention/feature visualization**: Understand what model learns

### Step 5.3: Root Cause Identification
Based on diagnostics, categorize findings:
1. **Critical**: Must fix for reasonable performance
2. **Important**: Significant impact on performance
3. **Minor**: Nice to have improvements
4. **Architectural**: Fundamental design limitations

**Checkpoint**: Present diagnostic findings and prioritize issues.

---

## Phase 6: Improvement Recommendations

**Goal**: Suggest actionable improvements at both theoretical and code levels.

### Step 6.1: Theoretical Improvements

**Model Architecture:**
- Suggest modifications to better handle your data characteristics
- Propose architectural innovations based on recent literature
- Recommend regularization strategies

**Loss Function:**
- Suggest modifications to loss for your specific task
- Propose auxiliary losses for better learning signals
- Recommend loss scaling/balancing strategies

**Training Strategy:**
- Curriculum learning approaches
- Transfer learning / pretraining strategies
- Multi-task learning opportunities

### Step 6.2: Code-Level Improvements

**Implementation Fixes:**
```python
# Template for suggesting code changes
## Issue: [description of problem]
## Location: [file.py:line_number]
## Current:
[code snippet]
## Suggested:
[improved code snippet]
## Rationale:
[why this helps]
```

**Optimization:**
- Memory efficiency improvements
- Computational speedup opportunities
- Numerical stability enhancements

**Code Quality:**
- Better error handling
- More informative logging
- Reproducibility improvements (seeding, checkpointing)

### Step 6.3: Domain-Specific Recommendations (scRNA-seq)

**Data Handling:**
- Batch effect correction integration
- Appropriate normalization for your data
- Handling of dropout/zero-inflation
- Temporal alignment for multi-timepoint data

**Biological Plausibility:**
- Gene pathway integration
- Cell type hierarchy awareness
- Uncertainty quantification

### Step 6.4: Experimental Validation Plan

Propose experiments to validate improvements:
1. **Ablation study design**: Test each improvement independently
2. **Comparison protocol**: Fair comparison with baselines
3. **Statistical significance**: Proper statistical testing
4. **Reproducibility checklist**: Seeds, multiple runs, error bars

**Checkpoint**: Review improvement recommendations and prioritize implementation.

---

## Usage

Invoke this skill with:
```
/deep-method-understanding
```

The agent will guide you through each phase interactively, asking clarifying questions and providing checkpoints for your review.

## Tips for Effective Use

1. **Start with NotebookLM**: Feed the manuscript first, then expand to supplements and related papers
2. **Iterate on understanding**: Don't rush through phases; revisit earlier phases as you learn more
3. **Document everything**: Keep notes on findings at each phase
4. **Test incrementally**: Run small experiments to validate understanding before full benchmarks
5. **Ask questions**: Use the interactive nature to clarify unclear concepts

## Resources

- NotebookLM: [Your notebook link]
- GitHub Repo: [Your repo path]
- Annotated Deep Learning Papers: https://nn.labml.ai/
- Papers with Code: https://paperswithcode.com/
