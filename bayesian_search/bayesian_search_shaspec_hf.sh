#!/bin/bash

# ============================================================================
# SHASPEC BAYESIAN OPTIMIZATION SEARCH - HF CHEXPERT VIT + PHENOTYPE-9
# ============================================================================

SEARCH_FOLDS=(1)

MODEL="shaspec"
TASK="phenotype"
GPU="0,1,2"

PRETRAINED=true
USE_DEMOGRAPHICS=false
CROSS_EVAL=""
MATCHED=false

N_CALLS=20
N_INITIAL_POINTS=5
ACQUISITION_FUNC="gp_hedge"
N_JOBS=8

RESUME_FROM_CHECKPOINT=false
CHECKPOINT_FILE=""
CLEANUP_ONLY=false

LR_FIXED=0.0001
BATCH_SIZE_FIXED=16
EPOCHS_VALUES=(50)
PATIENCE_VALUES=(10)
SEEDS=(42 123 1234)

INPUT_DIM_VALUES=(498)
NUM_CLASSES_VALUES=(9)
USE_PHENOTYPE9_FIXED=true

EHR_ENCODER_FIXED="transformer"
CXR_ENCODER_FIXED="hf_chexpert_vit"
HF_MODEL_ID_FIXED="codewithdark/vit-chest-xray"
FREEZE_VIT_FIXED=true
BIAS_TUNE_FIXED=false
PARTIAL_LAYERS_FIXED=0

DIM_FIXED=256
DROPOUT_FIXED=0.2
WEIGHT_STD_FIXED=true
EHR_N_HEAD_FIXED=4
EHR_N_LAYERS_FIXED=1
EHR_LSTM_BIDIRECTIONAL_FIXED=true
EHR_LSTM_NUM_LAYERS_FIXED=1
NHEAD_FIXED=4
NUM_LAYERS_FIXED=1
MAX_SEQ_LEN_FIXED=500

ALPHA_MIN=0.01
ALPHA_MAX=0.1
BETA_MIN=0.005
BETA_MAX=0.2

# ============================================================================
# SCRIPT IMPLEMENTATION
# ============================================================================

generate_results_dir() {
    local model=$1
    local task=$2
    local use_demographics=$3
    local cross_eval=$4
    local matched=$5
    local pretrained=$6

    local demographic_str="no_demo"
    [ "$use_demographics" = "true" ] && demographic_str="demo"

    local matched_str="full"
    [ "$matched" = "true" ] && matched_str="matched"

    local pretrained_str="no_pretrained"
    [ "$pretrained" = "true" ] && pretrained_str="pretrained"

    local cross_eval_str="standard"
    [ -n "$cross_eval" ] && cross_eval_str="$cross_eval"

    local results_dirname="${model}_${task}-${demographic_str}-${cross_eval_str}-${matched_str}-${pretrained_str}_bayesian_search_results"
    echo "/hdd/bayesian_search_experiments/${model}/${task}/lightning_logs/${results_dirname}"
}

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR=$(generate_results_dir "$MODEL" "$TASK" "$USE_DEMOGRAPHICS" "$CROSS_EVAL" "$MATCHED" "$PRETRAINED")
LOG_FILE="${RESULTS_DIR}/bayesian_search_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$RESULTS_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

create_bayesian_optimizer() {
cat > "${RESULTS_DIR}/bayesian_optimizer.py" << 'PYEOF'
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from skopt import dump, gp_minimize, load
    from skopt.space import Real
except ImportError:
    print("Installing scikit-optimize...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-optimize"])
    from skopt import dump, gp_minimize, load
    from skopt.space import Real


class BayesianShaSpecOptimizer:
    def __init__(self, config):
        self.config = config
        self.results_dir = config["results_dir"]
        self.log_file = config["log_file"]
        self.iteration = 0
        self.best_score = -np.inf
        self.best_params = None
        self.best_iteration = None
        self.results_data = []
        self.previous_result = None
        self.already_run_params = set()

        if config.get("resume_from_checkpoint", False):
            checkpoint_file = config.get("checkpoint_file") or os.path.join(self.results_dir, "bayesian_optimization_result.pkl")
            if os.path.exists(checkpoint_file):
                try:
                    self.previous_result = load(checkpoint_file)
                    self.log(f"Loaded checkpoint from: {checkpoint_file}")
                except Exception as exc:
                    self.log(f"Failed to load checkpoint {checkpoint_file}: {exc}")

        self.dimensions = [
            Real(config["alpha_min"], config["alpha_max"], name="alpha"),
            Real(config["beta_min"], config["beta_max"], name="beta"),
        ]
        self.dimension_names = [dim.name for dim in self.dimensions]
        self.scan_existing_experiments()

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"[{timestamp}] {message}"
        print(text)
        with open(self.log_file, "a") as f:
            f.write(text + "\n")

    def get_lightning_logs_root(self):
        return os.path.dirname(self.results_dir)

    def _params_key(self, params_dict):
        return (
            round(float(params_dict["alpha"]), 8),
            round(float(params_dict["beta"]), 8),
        )

    def _resolve_seed_exp_dir(self, exp_name, seed):
        seed_dir = os.path.join(self.results_dir, f"{exp_name}_seed{seed}")
        return seed_dir if os.path.isdir(seed_dir) else None

    def extract_version_name_from_seed_log(self, exp_name, seed):
        seed_dir = self._resolve_seed_exp_dir(exp_name, seed)
        if not seed_dir:
            return None

        seed_log = os.path.join(seed_dir, "output.log")
        if not os.path.exists(seed_log):
            return None

        try:
            with open(seed_log, "r") as f:
                content = f.read()
            match = re.search(r"in the ver_name (.+)", content)
            if match:
                return match.group(1).strip()
        except Exception as exc:
            self.log(f"Failed to parse version name from {seed_log}: {exc}")
        return None

    def get_real_training_dir(self, exp_name, seed):
        version_name = self.extract_version_name_from_seed_log(exp_name, seed)
        if not version_name:
            return None
        candidate = os.path.join(self.get_lightning_logs_root(), version_name)
        return candidate if os.path.isdir(candidate) else None

    def scan_existing_experiments(self):
        csv_file = os.path.join(self.results_dir, "results_summary.csv")
        if not os.path.exists(csv_file):
            return

        try:
            df = pd.read_csv(csv_file)
        except Exception as exc:
            self.log(f"Failed to read existing results_summary.csv: {exc}")
            return

        if df.empty:
            return

        self.results_data = df.to_dict("records")
        self.iteration = len(self.results_data)

        for _, row in df.iterrows():
            try:
                self.already_run_params.add((round(float(row["alpha"]), 8), round(float(row["beta"]), 8)))
            except Exception:
                continue

        if "PRAUC_mean" in df.columns and df["PRAUC_mean"].notna().any():
            best_row = df.loc[df["PRAUC_mean"].idxmax()]
            self.best_score = float(best_row["PRAUC_mean"])
            self.best_iteration = int(best_row["iteration"])
            self.best_params = {
                "alpha": float(best_row["alpha"]),
                "beta": float(best_row["beta"]),
            }

        self.log(f"Found {len(self.already_run_params)} existing parameter combination(s)")
        if self.best_iteration is not None:
            self.log(f"Existing best iteration: {self.best_iteration}, PRAUC={self.best_score:.4f}")

    def persist_results_summary(self):
        if not self.results_data:
            return
        df = pd.DataFrame(self.results_data)
        if "PRAUC_mean" in df.columns and df["PRAUC_mean"].notna().any():
            best_prauc = df["PRAUC_mean"].max()
            df["is_best"] = df["PRAUC_mean"] == best_prauc
        csv_file = os.path.join(self.results_dir, "results_summary.csv")
        df.to_csv(csv_file, index=False)

    def delete_iteration_checkpoints(self, exp_name, iteration_label=None):
        deleted_count = 0
        for seed in self.config["seeds"]:
            training_dir = self.get_real_training_dir(exp_name, seed)
            if not training_dir:
                self.log(f"   No training dir found for iter {iteration_label}, seed {seed}")
                continue
            checkpoint_dir = os.path.join(training_dir, "checkpoints")
            if os.path.exists(checkpoint_dir):
                try:
                    shutil.rmtree(checkpoint_dir)
                    deleted_count += 1
                    self.log(f"   Deleted checkpoints for iter {iteration_label}, seed {seed}: {checkpoint_dir}")
                except Exception as exc:
                    self.log(f"   Failed to delete {checkpoint_dir}: {exc}")
        return deleted_count

    def cleanup_existing_checkpoints_only(self):
        self.log("=" * 60)
        self.log("CLEANUP-ONLY MODE")
        self.log("=" * 60)
        if not self.results_data or self.best_iteration is None:
            self.log("No results/best iteration available; nothing to clean")
            return

        deleted_count = 0
        kept_count = 0
        for result in self.results_data:
            iteration = int(result["iteration"])
            if iteration == self.best_iteration:
                kept_count += 1
                continue
            deleted = self.delete_iteration_checkpoints(result["experiment_name"], iteration_label=iteration)
            if deleted > 0:
                deleted_count += 1

        self.log(f"Deleted {deleted_count} non-best iteration(s); kept {kept_count} best iteration(s)")
        self.save_best_iteration_detailed_metrics()
        self.export_best_experiment()

    def extract_metrics_from_log(self, log_file):
        try:
            with open(log_file, "r") as f:
                content = f.read()
        except Exception:
            return {}

        metrics = {}
        metric_section = content
        marker = "Testing DataLoader 0"
        if marker in content:
            metric_section = content.split(marker, 1)[1]

        patterns = {
            "PRAUC": r"overall/PRAUC:\s*([0-9]+\.[0-9]+)",
            "ROC_AUC": r"overall/ROC_AUC:\s*([0-9]+\.[0-9]+)",
            "F1_macro": r"overall/F1_macro:\s*([0-9]+\.[0-9]+)",
            "F1_weighted": r"overall/F1_weighted:\s*([0-9]+\.[0-9]+)",
            "Precision_macro": r"overall/Precision_macro:\s*([0-9]+\.[0-9]+)",
            "Precision_weighted": r"overall/Precision_weighted:\s*([0-9]+\.[0-9]+)",
            "Recall_macro": r"overall/Recall_macro:\s*([0-9]+\.[0-9]+)",
            "Recall_weighted": r"overall/Recall_weighted:\s*([0-9]+\.[0-9]+)",
        }

        for metric_name, pattern in patterns.items():
            matches = re.findall(pattern, metric_section)
            if not matches:
                matches = re.findall(pattern, content)
            metrics[metric_name] = float(matches[0]) if matches else None

        return metrics

    def _build_exp_base(self, params_dict, fold):
        return (
            f"{self.config['model']}_fold{fold}_"
            f"a{float(params_dict['alpha']):.4f}_"
            f"b{float(params_dict['beta']):.4f}"
        )

    def run_experiment_with_seeds(self, params_dict, fold):
        self.iteration += 1
        exp_base = self._build_exp_base(params_dict, fold)
        self.log(f"Starting Bayesian iteration {self.iteration}: {exp_base}")

        gpu_list = [int(g.strip()) for g in str(self.config["gpus"]).split(",") if g.strip()]
        if not gpu_list:
            gpu_list = [0]
        num_gpus = len(gpu_list)
        self.log(f"Running {len(self.config['seeds'])} seeds in parallel across GPUs: {gpu_list}")

        seed_tasks = []
        for i, seed in enumerate(self.config["seeds"]):
            gpu_id = gpu_list[i % num_gpus]
            seed_dir = os.path.join(self.results_dir, f"{exp_base}_seed{seed}")
            os.makedirs(seed_dir, exist_ok=True)

            cmd = [
                sys.executable, "../main.py",
                "--model", self.config["model"],
                "--mode", "train",
                "--task", self.config["task"],
                "--fold", str(fold),
                "--gpu", str(gpu_id),
                "--lr", str(self.config["lr_fixed"]),
                "--batch_size", str(self.config["batch_size_fixed"]),
                "--epochs", str(self.config["epochs"]),
                "--patience", str(self.config["patience"]),
                "--input_dim", str(self.config["input_dim"]),
                "--num_classes", str(self.config["num_classes"]),
                "--ehr_encoder", self.config["ehr_encoder_fixed"],
                "--cxr_encoder", self.config["cxr_encoder_fixed"],
                "--hf_model_id", self.config["hf_model_id_fixed"],
                "--freeze_vit", str(self.config["freeze_vit_fixed"]).lower(),
                "--bias_tune", str(self.config["bias_tune_fixed"]).lower(),
                "--partial_layers", str(self.config["partial_layers_fixed"]),
                "--dim", str(self.config["dim_fixed"]),
                "--dropout", str(self.config["dropout_fixed"]),
                "--ehr_n_head", str(self.config["ehr_n_head_fixed"]),
                "--ehr_n_layers", str(self.config["ehr_n_layers_fixed"]),
                "--ehr_num_layers", str(self.config["ehr_lstm_num_layers_fixed"]),
                "--nhead", str(self.config["nhead_fixed"]),
                "--num_layers", str(self.config["num_layers_fixed"]),
                "--max_seq_len", str(self.config["max_seq_len_fixed"]),
                "--alpha", str(float(params_dict["alpha"])),
                "--beta", str(float(params_dict["beta"])),
                "--seed", str(seed),
                "--log_dir", f"/hdd/bayesian_search_experiments/{self.config['model']}/{self.config['task']}",
            ]

            if self.config["pretrained"]:
                cmd.append("--pretrained")
            if self.config["matched"]:
                cmd.append("--matched")
            if self.config["use_demographics"]:
                cmd.append("--use_demographics")
            if self.config["use_phenotype9_fixed"]:
                cmd.append("--use_phenotype9")
            if self.config["ehr_lstm_bidirectional_fixed"]:
                cmd.append("--ehr_bidirectional")
            if self.config["weight_std_fixed"]:
                cmd.append("--weight_std")
            if self.config["cross_eval"]:
                cmd.extend(["--cross_eval", self.config["cross_eval"]])

            seed_tasks.append((seed, seed_dir, cmd))

        def run_single_seed(seed, seed_dir, cmd):
            output_log = os.path.join(seed_dir, "output.log")
            try:
                with open(output_log, "w") as f:
                    result = subprocess.run(
                        cmd,
                        cwd=self.config["base_dir"],
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        timeout=None,
                    )
                return seed, result.returncode == 0, output_log, None
            except Exception as exc:
                return seed, False, output_log, str(exc)

        all_metrics = []
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = {
                executor.submit(run_single_seed, seed, seed_dir, cmd): seed
                for seed, seed_dir, cmd in seed_tasks
            }
            for future in as_completed(futures):
                seed = futures[future]
                seed, success, output_log, error = future.result()
                if not success:
                    self.log(f"Seed {seed} failed: {error or 'non-zero return code'}")
                    continue
                metrics = self.extract_metrics_from_log(output_log)
                if metrics and any(v is not None for v in metrics.values()):
                    all_metrics.append(metrics)
                    metric_text = " | ".join(
                        f"{k}={v:.4f}" if v is not None else f"{k}=N/A"
                        for k, v in metrics.items()
                    )
                    self.log(f"Seed {seed}: {metric_text}")
                else:
                    self.log(f"Seed {seed}: metrics not found in log")

        if not all_metrics:
            self.log(f"No valid metrics found for iteration {self.iteration}")
            self.delete_iteration_checkpoints(exp_base, iteration_label=self.iteration)
            return -1.0, -1.0

        metric_names = [
            "PRAUC", "ROC_AUC", "F1_macro", "F1_weighted",
            "Precision_macro", "Precision_weighted", "Recall_macro", "Recall_weighted",
        ]
        metric_stats = {}
        for metric_name in metric_names:
            values = [m.get(metric_name) for m in all_metrics if m.get(metric_name) is not None]
            metric_stats[f"{metric_name}_mean"] = float(np.mean(values)) if values else None
            metric_stats[f"{metric_name}_std"] = float(np.std(values)) if values else None

        result_parts = []
        for metric_name in metric_names:
            mean_val = metric_stats[f"{metric_name}_mean"]
            std_val = metric_stats[f"{metric_name}_std"]
            if mean_val is None:
                result_parts.append(f"{metric_name}: N/A")
            else:
                result_parts.append(f"{metric_name}: {mean_val:.4f}±{std_val:.4f}")
        self.log(f"Iteration {self.iteration} - " + " | ".join(result_parts))

        prauc_mean = metric_stats["PRAUC_mean"]
        is_new_best = False
        if prauc_mean is not None and prauc_mean > self.best_score:
            self.best_score = prauc_mean
            self.best_params = {
                "alpha": float(params_dict["alpha"]),
                "beta": float(params_dict["beta"]),
            }
            self.best_iteration = self.iteration
            is_new_best = True
            self.log(f"New best PRAUC: {self.best_score:.4f}")

        result_data = {
            "iteration": self.iteration,
            "experiment_name": exp_base,
            "fold": fold,
            "lr_fixed": self.config["lr_fixed"],
            "task": self.config["task"],
            "use_demographics": self.config["use_demographics"],
            "cross_eval": self.config["cross_eval"],
            "pretrained": self.config["pretrained"],
            "cxr_encoder": self.config["cxr_encoder_fixed"],
            "hf_model_id": self.config["hf_model_id_fixed"],
            "use_phenotype9": self.config["use_phenotype9_fixed"],
            "alpha": float(params_dict["alpha"]),
            "beta": float(params_dict["beta"]),
            **metric_stats,
            "all_metrics": json.dumps(all_metrics),
            "is_best": is_new_best,
        }
        self.results_data.append(result_data)
        self.already_run_params.add(self._params_key(params_dict))
        self.persist_results_summary()

        if is_new_best:
            for previous_result in self.results_data[:-1]:
                if int(previous_result["iteration"]) != self.best_iteration:
                    self.delete_iteration_checkpoints(
                        previous_result["experiment_name"],
                        iteration_label=int(previous_result["iteration"]),
                    )
        else:
            self.delete_iteration_checkpoints(exp_base, iteration_label=self.iteration)

        return prauc_mean if prauc_mean is not None else -1.0, metric_stats["F1_macro_mean"] or -1.0

    def objective_function(self, params):
        params_dict = dict(zip(self.dimension_names, params))
        params_key = self._params_key(params_dict)
        if params_key in self.already_run_params:
            self.log(f"Skipping already-run params: {params_dict}")
            return -0.5

        scores = []
        for fold in self.config["search_folds"]:
            score, _ = self.run_experiment_with_seeds(params_dict, fold)
            scores.append(score)
        return -float(np.mean(scores))

    def generate_convergence_plot(self, result):
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8), facecolor="white")
            scores = [-y for y in result.func_vals]
            best_scores = [max(scores[: i + 1]) for i in range(len(scores))]
            alphas = [x[0] for x in result.x_iters] if hasattr(result, "x_iters") else []
            betas = [x[1] for x in result.x_iters] if hasattr(result, "x_iters") else []

            plt.subplot(2, 2, 1)
            plt.plot(scores, "bo-", alpha=0.6, label="PRAUC")
            plt.plot(best_scores, "r-", linewidth=2, label="Best PRAUC")
            plt.xlabel("Iteration")
            plt.ylabel("PRAUC")
            plt.title("Bayesian Optimization Convergence")
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.subplot(2, 2, 2)
            plt.plot(alphas, "g-o", alpha=0.7)
            plt.xlabel("Iteration")
            plt.ylabel("alpha")
            plt.title("Alpha Exploration")
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 3)
            plt.plot(betas, "b-o", alpha=0.7)
            plt.xlabel("Iteration")
            plt.ylabel("beta")
            plt.title("Beta Exploration")
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 4)
            plt.scatter(alphas, betas, c=scores, cmap="viridis", alpha=0.8)
            plt.xlabel("alpha")
            plt.ylabel("beta")
            plt.title("Alpha/Beta vs PRAUC")
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = os.path.join(self.results_dir, "convergence_plot.png")
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            self.log(f"Saved convergence plot to: {plot_file}")
        except Exception as exc:
            self.log(f"Could not generate convergence plot: {exc}")

    def build_best_export_name(self, result_row):
        return (
            f"BEST_{self.config['model']}_{self.config['task']}_"
            f"fold{int(result_row['fold'])}_"
            f"a{float(result_row['alpha']):.4f}_"
            f"b{float(result_row['beta']):.4f}_"
            f"prauc{float(result_row['PRAUC_mean']):.4f}"
        )

    def export_best_experiment(self):
        if self.best_iteration is None or not self.results_data:
            self.log("No best iteration available; skipping BEST export")
            return

        best_rows = [row for row in self.results_data if int(row["iteration"]) == self.best_iteration]
        if not best_rows:
            self.log("Best iteration not found in results data; skipping BEST export")
            return

        best_row = best_rows[-1]
        target_dir = os.path.join(self.results_dir, self.build_best_export_name(best_row))
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)

        exported_seeds = {}
        for seed in self.config["seeds"]:
            source_dir = self.get_real_training_dir(best_row["experiment_name"], seed)
            if not source_dir:
                self.log(f"Could not resolve best training dir for seed {seed}")
                continue
            seed_target_dir = os.path.join(target_dir, f"seed{seed}")
            shutil.copytree(source_dir, seed_target_dir)
            exported_seeds[str(seed)] = {
                "source_dir": source_dir,
                "exported_dir": seed_target_dir,
            }

        if not exported_seeds:
            shutil.rmtree(target_dir)
            self.log("Could not export any best seed directories")
            return

        metadata = {
            "best_iteration": int(self.best_iteration),
            "best_experiment_name": best_row["experiment_name"],
            "best_score": float(self.best_score),
            "best_params": {
                "alpha": float(best_row["alpha"]),
                "beta": float(best_row["beta"]),
            },
            "exported_seeds": exported_seeds,
        }
        metadata_file = os.path.join(target_dir, "best_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        self.log(f"Exported BEST experiment to: {target_dir}")

    def save_best_iteration_detailed_metrics(self):
        if self.best_iteration is None:
            self.log("No best iteration found; skipping detailed metrics")
            return

        best_rows = [row for row in self.results_data if int(row["iteration"]) == self.best_iteration]
        if not best_rows:
            self.log("Could not find result row for best iteration")
            return

        best_row = best_rows[-1]
        all_seed_metrics = {}
        seeds_found = []

        for seed in self.config["seeds"]:
            training_dir = self.get_real_training_dir(best_row["experiment_name"], seed)
            if not training_dir:
                self.log(f"Could not resolve training dir for seed {seed}")
                continue
            metrics_file = os.path.join(training_dir, "test_set_results.yaml")
            if not os.path.exists(metrics_file):
                self.log(f"Missing test_set_results.yaml for seed {seed}: {metrics_file}")
                continue
            try:
                import yaml
                with open(metrics_file, "r") as f:
                    metrics = yaml.safe_load(f)
                if metrics:
                    all_seed_metrics[seed] = metrics
                    seeds_found.append(seed)
            except Exception as exc:
                self.log(f"Failed to load {metrics_file}: {exc}")

        if not all_seed_metrics:
            self.log("No best-iteration metrics loaded from any seed")
            return

        all_metric_keys = set()
        for seed_metrics in all_seed_metrics.values():
            all_metric_keys.update(seed_metrics.keys())

        detailed_results = []
        for metric_key in sorted(all_metric_keys):
            values = [seed_metrics[metric_key] for seed_metrics in all_seed_metrics.values() if metric_key in seed_metrics]
            if not values:
                continue
            mean_val = float(np.mean(values))
            std_val = float(np.std(values)) if len(values) > 1 else 0.0
            detailed_results.append({
                "metric": metric_key,
                "mean": mean_val,
                "std": std_val,
                "formatted": f"{mean_val:.4f} ± {std_val:.4f}",
                "seeds_count": len(values),
            })

        if not detailed_results:
            self.log("No detailed metrics to save")
            return

        csv_file = os.path.join(self.results_dir, "best_iteration_detailed_metrics.csv")
        pd.DataFrame(detailed_results).to_csv(csv_file, index=False)
        self.log(f"Saved best iteration detailed metrics to: {csv_file}")
        self.log(f"Loaded detailed metrics from seeds: {seeds_found}")

    def cleanup_final_checkpoints(self):
        self.log("=" * 60)
        self.log("FINAL CHECKPOINT CLEANUP")
        self.log("=" * 60)
        if not self.results_data or self.best_iteration is None:
            self.log("No results/best iteration available; skipping final cleanup")
            return

        deleted_count = 0
        kept_count = 0
        for result in self.results_data:
            iteration = int(result["iteration"])
            if iteration == self.best_iteration:
                kept_count += 1
                continue
            deleted = self.delete_iteration_checkpoints(result["experiment_name"], iteration_label=iteration)
            if deleted > 0:
                deleted_count += 1

        self.log(f"Deleted {deleted_count} non-best iteration(s); kept {kept_count} best iteration(s)")
        self.save_best_iteration_detailed_metrics()

    def save_best_params(self):
        best_params_file = os.path.join(self.results_dir, "best_params.txt")
        with open(best_params_file, "w") as f:
            f.write("ShaSpec Bayesian Optimization Best Parameters\n")
            f.write("=" * 50 + "\n")
            f.write(f"Best PRAUC: {self.best_score:.4f}\n")
            f.write(f"Total iterations: {len(self.results_data)}\n")
            if self.best_iteration is not None:
                f.write(f"Best iteration: {self.best_iteration}\n")
            f.write("\nBest Parameters:\n")
            if self.best_params:
                for key, value in self.best_params.items():
                    f.write(f"  {key}: {value}\n")
            f.write(f"Fixed CXR Encoder: {self.config['cxr_encoder_fixed']}\n")
            f.write(f"Fixed HF Model ID: {self.config['hf_model_id_fixed']}\n")
            f.write(f"Use Phenotype9: {self.config['use_phenotype9_fixed']}\n")
            f.write(f"Seeds used: {self.config['seeds']}\n")
        self.log(f"Saved best parameters to: {best_params_file}")

    def run_optimization(self):
        if self.config.get("cleanup_only", False):
            self.log("Cleanup-only mode enabled; skipping new experiments")
            self.cleanup_existing_checkpoints_only()
            self.save_best_params()
            return None

        self.log("Starting Bayesian Optimization for ShaSpec")
        self.log(f"Seeds: {self.config['seeds']}")
        self.log(f"GPU(s): {self.config['gpus']}")
        self.log(f"Search space: {self.dimension_names}")
        self.log(f"Fixed task setup: task={self.config['task']}, num_classes={self.config['num_classes']}, use_phenotype9={self.config['use_phenotype9_fixed']}")
        self.log(f"Fixed CXR encoder: {self.config['cxr_encoder_fixed']} ({self.config['hf_model_id_fixed']})")

        if self.previous_result is not None:
            remaining_calls = self.config["n_calls"] - len(self.previous_result.x_iters)
            if remaining_calls <= 0:
                self.log("Previous optimization already reached requested calls")
                self.save_best_params()
                self.export_best_experiment()
                self.cleanup_final_checkpoints()
                return self.previous_result

            self.log(f"Continuing optimization with {remaining_calls} remaining call(s)")
            from skopt import Optimizer

            optimizer = Optimizer(
                dimensions=self.dimensions,
                acq_func=self.config["acquisition_func"],
                n_initial_points=0,
                random_state=42,
            )
            for x, y in zip(self.previous_result.x_iters, self.previous_result.func_vals):
                optimizer.tell(x, y)

            for i in range(remaining_calls):
                next_x = optimizer.ask()
                next_y = self.objective_function(next_x)
                optimizer.tell(next_x, next_y)
                if (i + 1) % 5 == 0:
                    optimization_file = os.path.join(self.results_dir, "bayesian_optimization_result.pkl")
                    dump(optimizer, optimization_file)
                    self.log(f"Saved resume checkpoint at continuation step {i + 1}")

            result = optimizer
        else:
            result = gp_minimize(
                func=self.objective_function,
                dimensions=self.dimensions,
                n_calls=self.config["n_calls"],
                n_initial_points=self.config["n_initial_points"],
                acq_func=self.config["acquisition_func"],
                n_jobs=self.config["n_jobs"],
                random_state=42,
            )

        optimization_file = os.path.join(self.results_dir, "bayesian_optimization_result.pkl")
        dump(result, optimization_file)
        self.persist_results_summary()
        self.save_best_params()
        self.generate_convergence_plot(result)
        self.export_best_experiment()
        self.cleanup_final_checkpoints()

        self.log("=== BAYESIAN OPTIMIZATION COMPLETED ===")
        self.log(f"Best PRAUC found: {self.best_score:.4f}")
        self.log(f"Best parameters: {self.best_params}")
        return result


def main():
    config = {
        "results_dir": os.environ.get("RESULTS_DIR"),
        "log_file": os.environ.get("LOG_FILE"),
        "base_dir": os.environ.get("BASE_DIR"),
        "model": os.environ.get("MODEL", "shaspec"),
        "task": os.environ.get("TASK", "phenotype"),
        "gpus": os.environ.get("GPU", "0,1,2"),
        "epochs": int(os.environ.get("EPOCHS", 50)),
        "patience": int(os.environ.get("PATIENCE", 10)),
        "input_dim": int(os.environ.get("INPUT_DIM", 498)),
        "num_classes": int(os.environ.get("NUM_CLASSES", 9)),
        "pretrained": os.environ.get("PRETRAINED", "true").lower() == "true",
        "matched": os.environ.get("MATCHED", "false").lower() == "true",
        "use_demographics": os.environ.get("USE_DEMOGRAPHICS", "false").lower() == "true",
        "cross_eval": os.environ.get("CROSS_EVAL", ""),
        "search_folds": [int(x) for x in os.environ.get("SEARCH_FOLDS", "1").split(",")],
        "n_calls": int(os.environ.get("N_CALLS", 20)),
        "n_initial_points": int(os.environ.get("N_INITIAL_POINTS", 5)),
        "acquisition_func": os.environ.get("ACQUISITION_FUNC", "gp_hedge"),
        "n_jobs": int(os.environ.get("N_JOBS", 8)),
        "resume_from_checkpoint": os.environ.get("RESUME_FROM_CHECKPOINT", "false").lower() == "true",
        "checkpoint_file": os.environ.get("CHECKPOINT_FILE", ""),
        "cleanup_only": os.environ.get("CLEANUP_ONLY", "false").lower() == "true",
        "lr_fixed": float(os.environ.get("LR_FIXED", 0.0001)),
        "batch_size_fixed": int(os.environ.get("BATCH_SIZE_FIXED", 16)),
        "seeds": [int(x) for x in os.environ.get("SEEDS", "42,123,1234").split(",")],
        "use_phenotype9_fixed": os.environ.get("USE_PHENOTYPE9_FIXED", "true").lower() == "true",
        "ehr_encoder_fixed": os.environ.get("EHR_ENCODER_FIXED", "transformer"),
        "cxr_encoder_fixed": os.environ.get("CXR_ENCODER_FIXED", "hf_chexpert_vit"),
        "hf_model_id_fixed": os.environ.get("HF_MODEL_ID_FIXED", "codewithdark/vit-chest-xray"),
        "freeze_vit_fixed": os.environ.get("FREEZE_VIT_FIXED", "true").lower() == "true",
        "bias_tune_fixed": os.environ.get("BIAS_TUNE_FIXED", "false").lower() == "true",
        "partial_layers_fixed": int(os.environ.get("PARTIAL_LAYERS_FIXED", 0)),
        "dim_fixed": int(os.environ.get("DIM_FIXED", 256)),
        "dropout_fixed": float(os.environ.get("DROPOUT_FIXED", 0.2)),
        "weight_std_fixed": os.environ.get("WEIGHT_STD_FIXED", "true").lower() == "true",
        "ehr_n_head_fixed": int(os.environ.get("EHR_N_HEAD_FIXED", 4)),
        "ehr_n_layers_fixed": int(os.environ.get("EHR_N_LAYERS_FIXED", 1)),
        "ehr_lstm_bidirectional_fixed": os.environ.get("EHR_LSTM_BIDIRECTIONAL_FIXED", "true").lower() == "true",
        "ehr_lstm_num_layers_fixed": int(os.environ.get("EHR_LSTM_NUM_LAYERS_FIXED", 1)),
        "nhead_fixed": int(os.environ.get("NHEAD_FIXED", 4)),
        "num_layers_fixed": int(os.environ.get("NUM_LAYERS_FIXED", 1)),
        "max_seq_len_fixed": int(os.environ.get("MAX_SEQ_LEN_FIXED", 500)),
        "alpha_min": float(os.environ.get("ALPHA_MIN", 0.01)),
        "alpha_max": float(os.environ.get("ALPHA_MAX", 0.1)),
        "beta_min": float(os.environ.get("BETA_MIN", 0.005)),
        "beta_max": float(os.environ.get("BETA_MAX", 0.2)),
    }

    optimizer = BayesianShaSpecOptimizer(config)
    optimizer.run_optimization()
    print("ShaSpec Bayesian optimization completed successfully!")


if __name__ == "__main__":
    main()
PYEOF
}

main() {
    log "Starting ShaSpec Bayesian Optimization Search"
    log "Configuration: MODEL=$MODEL, TASK=$TASK, USE_DEMOGRAPHICS=$USE_DEMOGRAPHICS, CROSS_EVAL=$CROSS_EVAL, PRETRAINED=$PRETRAINED"
    log "Results directory: $RESULTS_DIR"
    log "Log file: $LOG_FILE"
    log "Search parameters: alpha=[$ALPHA_MIN, $ALPHA_MAX], beta=[$BETA_MIN, $BETA_MAX]"
    log "Fixed task setup: phenotype9 + hf_chexpert_vit + codewithdark/vit-chest-xray"

    create_bayesian_optimizer

    export RESULTS_DIR="$RESULTS_DIR"
    export LOG_FILE="$LOG_FILE"
    export BASE_DIR="$BASE_DIR"
    export MODEL="$MODEL"
    export TASK="$TASK"
    export GPU="$GPU"
    export EPOCHS="${EPOCHS_VALUES[0]}"
    export PATIENCE="${PATIENCE_VALUES[0]}"
    export INPUT_DIM="${INPUT_DIM_VALUES[0]}"
    export NUM_CLASSES="${NUM_CLASSES_VALUES[0]}"
    export PRETRAINED="$PRETRAINED"
    export MATCHED="$MATCHED"
    export USE_DEMOGRAPHICS="$USE_DEMOGRAPHICS"
    export CROSS_EVAL="$CROSS_EVAL"
    export SEARCH_FOLDS=$(IFS=,; echo "${SEARCH_FOLDS[*]}")
    export N_CALLS="$N_CALLS"
    export N_INITIAL_POINTS="$N_INITIAL_POINTS"
    export ACQUISITION_FUNC="$ACQUISITION_FUNC"
    export N_JOBS="$N_JOBS"
    export RESUME_FROM_CHECKPOINT="$RESUME_FROM_CHECKPOINT"
    export CHECKPOINT_FILE="$CHECKPOINT_FILE"
    export CLEANUP_ONLY="$CLEANUP_ONLY"
    export LR_FIXED="$LR_FIXED"
    export BATCH_SIZE_FIXED="$BATCH_SIZE_FIXED"
    export SEEDS=$(IFS=,; echo "${SEEDS[*]}")
    export USE_PHENOTYPE9_FIXED="$USE_PHENOTYPE9_FIXED"
    export EHR_ENCODER_FIXED="$EHR_ENCODER_FIXED"
    export CXR_ENCODER_FIXED="$CXR_ENCODER_FIXED"
    export HF_MODEL_ID_FIXED="$HF_MODEL_ID_FIXED"
    export FREEZE_VIT_FIXED="$FREEZE_VIT_FIXED"
    export BIAS_TUNE_FIXED="$BIAS_TUNE_FIXED"
    export PARTIAL_LAYERS_FIXED="$PARTIAL_LAYERS_FIXED"
    export DIM_FIXED="$DIM_FIXED"
    export DROPOUT_FIXED="$DROPOUT_FIXED"
    export WEIGHT_STD_FIXED="$WEIGHT_STD_FIXED"
    export EHR_N_HEAD_FIXED="$EHR_N_HEAD_FIXED"
    export EHR_N_LAYERS_FIXED="$EHR_N_LAYERS_FIXED"
    export EHR_LSTM_BIDIRECTIONAL_FIXED="$EHR_LSTM_BIDIRECTIONAL_FIXED"
    export EHR_LSTM_NUM_LAYERS_FIXED="$EHR_LSTM_NUM_LAYERS_FIXED"
    export NHEAD_FIXED="$NHEAD_FIXED"
    export NUM_LAYERS_FIXED="$NUM_LAYERS_FIXED"
    export MAX_SEQ_LEN_FIXED="$MAX_SEQ_LEN_FIXED"
    export ALPHA_MIN="$ALPHA_MIN"
    export ALPHA_MAX="$ALPHA_MAX"
    export BETA_MIN="$BETA_MIN"
    export BETA_MAX="$BETA_MAX"

    cd "$BASE_DIR"
    python3 "${RESULTS_DIR}/bayesian_optimizer.py"

    if [ $? -eq 0 ]; then
        log "ShaSpec Bayesian optimization completed successfully"
        log "Results saved to: $RESULTS_DIR"
    else
        log "ShaSpec Bayesian optimization failed"
        exit 1
    fi
}

cleanup() {
    log "ShaSpec Bayesian search interrupted by user"
    exit 1
}

trap cleanup SIGINT SIGTERM
main "$@"
