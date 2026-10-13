import torch
from data.datamodule import MultiViewDataModule
from data.datasets.hw import HWDataset
from utils import MetricsCalculator, StatisticalSignificanceTester

# 1. Setup Data (Original HW Dataset from /home/phd/datasets/)
datamodule = MultiViewDataModule(
    dataset_class=HWDataset,
    root_dir="/home/phd/datasets/hw",
    batch_size=256,
    num_workers=4
)
datamodule.setup()

num_classes = 10
metrics_calc = MetricsCalculator(num_classes=num_classes)
tester = StatisticalSignificanceTester(alpha=0.05)

# 2. Define two deterministic baselines that use the original data (no random values)
def baseline_view_0(views):
    # Deterministic prediction based on the first view's first feature
    return (views[0][:, 0] * 100).long() % num_classes

def baseline_view_1(views):
    # Deterministic prediction based on the second view's first feature
    return (views[1][:, 0] * 100).long() % num_classes

# 3. Collect all views and targets from the original strict inductive test split (80%)
test_subset = datamodule.test_dataloader().dataset

all_views_list = []
all_targets_list = []
for views, target in test_subset:
    all_views_list.append(views)
    all_targets_list.append(target)

# Stack into tensors
num_views = len(all_views_list[0])
all_views = [torch.stack([v[i] for v in all_views_list]) for i in range(num_views)]
all_targets = torch.tensor(all_targets_list)

# 4. Simulate 5 independent evaluations by splitting the test set into 5 chunks
# (This mimics evaluating across 5 independent seeds/folds without using random splits)
chunk_size = len(all_targets) // 5
proposed_scores = []
baseline_scores = []

for i in range(5):
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size if i < 4 else len(all_targets)
    
    chunk_views = [v[start_idx:end_idx] for v in all_views]
    chunk_targets = all_targets[start_idx:end_idx]
    
    # Compute metrics for "Proposed" (View 0 baseline)
    preds_prop = baseline_view_0(chunk_views)
    metrics_prop = metrics_calc.compute_all_metrics(chunk_targets, preds_prop)
    proposed_scores.append(metrics_prop['Macro-F1'])
    
    # Compute metrics for "Baseline" (View 1 baseline)
    preds_base = baseline_view_1(chunk_views)
    metrics_base = metrics_calc.compute_all_metrics(chunk_targets, preds_base)
    baseline_scores.append(metrics_base['Macro-F1'])

proposed_scores = torch.tensor(proposed_scores)
baseline_scores = torch.tensor(baseline_scores)

# 5. Perform Wilcoxon Signed-Rank Test
w_stat, p_value = tester.wilcoxon_signed_rank_test(proposed_scores, baseline_scores)
print(f"--- Wilcoxon Signed-Rank Test ---")
print(f"W+ Statistic: {w_stat:.2f}")
print(f"Uncorrected p-value: {p_value:.4f}\n")

# 6. Apply Bonferroni Correction (assuming we tested against 3 baselines across 2 datasets = 6 tests)
p_values = [p_value] * 6 # Dummy list for demonstration of multiple tests
corrected_p, adj_alpha = tester.bonferroni_correction(p_values)
print(f"--- Bonferroni Correction ---")
print(f"Number of tests (m): {len(p_values)}")
print(f"Adjusted alpha (alpha_adj): {adj_alpha:.4f}")
print(f"Corrected p-values: {[f'{p:.4f}' for p in corrected_p]}")