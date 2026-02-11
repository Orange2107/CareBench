import os

def get_version_name(args):
    args_dict = vars(args) if not isinstance(args, dict) else args

    # Core parameters to include in version name
    keys_to_keep = ["model", "fusion_type", "task", "fold", "batch_size", "lr", "patience", "epochs", "dropout", "seed", "align", "pretrained"]

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
    param_strs = [f"{k}_{args_dict[k]}" for k in keys_to_keep if k in args_dict]

    # Construct version name by joining parameters with hyphens
    version_name = f"{args_dict['model'].upper()}-" + "-".join(param_strs)

    if "log_dir" in args_dict and args_dict["log_dir"]:
        log_dir = args_dict["log_dir"]
    else:
        # Automatically construct log directory path
        log_dir = os.path.join("./experiments", f"{args_dict['model']}", args_dict["task"])

    return log_dir, version_name