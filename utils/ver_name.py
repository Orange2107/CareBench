import os
import hashlib
import string


def _format_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # Keep enough precision for experiment identity while avoiding
        # path blow-up from long floating-point strings.
        return f"{value:.4g}"
    return str(value)


EXPERIMENT_NAME_TEMPLATES = {
    "healnet":  "{model}_fold{fold}_d{depth}_fb{num_freq_bands}_mf{max_freq}_seed{seeds}",

    "crossvpt": "{model}_fold{fold}_seed{seed}_npt{num_prompt_tokens}_ptd{prompt_token_dropout}_pns{prompt_noise_std}",

    "drfuse": "{model}_fold{fold}_lds{lambda_disentangle_shared:.4f}_lde{lambda_disentangle_ehr:.4f}_ldc{lambda_disentangle_cxr:.4f}_lpe{lambda_pred_ehr:.4f}_lpc{lambda_pred_cxr:.4f}_lps{lambda_pred_shared:.4f}_laa{lambda_attn_aux:.4f}_seed{seeds}",

    "medfuse": "{model}_fold{fold}_seed{seed}_al{align:.4f}_fhd{fusion_lstm_hidden_dim}_fll{fusion_lstm_layers}_fld{fusion_lstm_dropout:.4f}",

    "flexmoe": "{model}_fold{fold}_ne{num_experts}_nr{num_routers}_tk{top_k}_glw{gate_loss_weight:.4f}_seed{seeds}",

    "umse": "{model}_fold{fold}_bn{bottlenecks_n}_nl{num_layers}_nh{num_heads}_dm{d_model}_do{dropout}_seed{seed}",

    "shaspec": "{model}_fold{fold}_a{alpha:.4f}_b{beta:.4f}_seed{seeds}",
    "smil": "{model}_fold{fold}_il{inner_loop}_mc{mc_size}_lri{lr_inner:.4f}_a{alpha:.4f}_b{beta:.4f}_t{temperature:.4f}_seed{seeds}",
    "latefusion": "{model}_fold{fold}_seed{seed}",
}
DEFAULT_EXPERIMENT_TEMPLATE = "{model}_fold{fold}_seed{seed}"


def _normalize_seed_like(value):
    if isinstance(value, (list, tuple)):
        return "-".join(str(v) for v in value)
    return str(value)


def _build_template_context(args_dict):
    context = {}
    for key, value in args_dict.items():
        if key == "seed":
            context["seed"] = _normalize_seed_like(value)
            if "seeds" not in context:
                context["seeds"] = context["seed"]
        elif key == "seeds":
            if value is not None:
                context["seeds"] = _normalize_seed_like(value)
        elif isinstance(value, (list, tuple)):
            context[key] = _normalize_seed_like(value)
        else:
            # Keep numeric values as numbers so format specs like :.4f work.
            context[key] = value

    context.setdefault("model", str(args_dict.get("model", "")))
    context.setdefault("iteration", str(args_dict.get("iteration", args_dict.get("iter", "na"))))
    context.setdefault("seeds", context.get("seed", "na"))
    return context


def _try_render_template(args_dict):
    model_key = str(args_dict.get("model", "")).lower()
    template = EXPERIMENT_NAME_TEMPLATES.get(model_key, DEFAULT_EXPERIMENT_TEMPLATE)
    if not template:
        return None

    context = _build_template_context(args_dict)
    formatter = string.Formatter()
    required_fields = [field_name for _, field_name, _, _ in formatter.parse(template) if field_name]
    if any(field not in context for field in required_fields):
        return None

    return template.format(**context)


def _cap_version_name(version_name):
    max_component_len = 220
    if len(version_name) > max_component_len:
        digest = hashlib.sha1(version_name.encode("utf-8")).hexdigest()[:10]
        keep = max_component_len - len("-h") - len(digest)
        version_name = f"{version_name[:keep]}-h{digest}"
    return version_name


def get_version_name(args):
    args_dict = vars(args) if not isinstance(args, dict) else args
    custom_experiment_name = args_dict.get("experiment_name")
    if custom_experiment_name:
        if "log_dir" in args_dict and args_dict["log_dir"]:
            log_dir = args_dict["log_dir"]
        else:
            log_dir = os.path.join("./experiments", f"{args_dict['model']}", args_dict["task"])
        return log_dir, _cap_version_name(str(custom_experiment_name))

    template_version_name = _try_render_template(args_dict)
    if template_version_name:
        if "log_dir" in args_dict and args_dict["log_dir"]:
            log_dir = args_dict["log_dir"]
        else:
            log_dir = os.path.join("./experiments", f"{args_dict['model']}", args_dict["task"])
        return log_dir, _cap_version_name(template_version_name)

    # Core parameters to include in version name
    keys_to_keep = [
        "model",
        "fusion_type",
        "task",
        "fold",
        "batch_size",
        "lr",
        "patience",
        "epochs",
        "dropout",
        "seed",
        "align",
        "pretrained",
    ]

    # Allow each model config to declare extra hyperparameters that should be
    # reflected in the experiment name to avoid collisions during search.
    tracked_hparams = args_dict.get("tracked_hparams", []) or []
    if isinstance(tracked_hparams, str):
        tracked_hparams = [tracked_hparams]
    for key in tracked_hparams:
        if key not in keys_to_keep:
            keys_to_keep.append(key)

    # Handle data configuration parameters
    data_config = ""
    if hasattr(args, 'cross_eval') and args.cross_eval:
        data_config = f"cross_{args.cross_eval}"
    elif hasattr(args, 'matched') and args.matched:
        data_config = "matched"
    else:
        data_config = "full"
    
    # Add demographic configuration
    if hasattr(args, 'use_demographics') and args.use_demographics:
        data_config += "_demo"
    
    # Add data configuration to version name
    keys_to_keep.append("data_config")
    args_dict["data_config"] = data_config

    original_name = args_dict["model"]
    if hasattr(args, 'unimodal_loss') and args.unimodal_loss:
        args_dict['model'] = f"{original_name}_uniloss"
    # Join parameter strings with underscores
    param_strs = [f"{k}_{_format_value(args_dict[k])}" for k in keys_to_keep if k in args_dict]

    # Construct version name by joining parameters with hyphens
    version_name = f"{args_dict['model'].upper()}-" + "-".join(param_strs)
    version_name = _cap_version_name(version_name)

    if "log_dir" in args_dict and args_dict["log_dir"]:
        log_dir = args_dict["log_dir"]
    else:
        # Automatically construct log directory path
        log_dir = os.path.join("./experiments", f"{args_dict['model']}", args_dict["task"])

    return log_dir, version_name
