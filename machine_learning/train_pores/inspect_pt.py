import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="Inspect a PyTorch .pt file")
    parser.add_argument("--model-path", required=True, help="Path to .pt file")
    args = parser.parse_args()

    obj = torch.load(args.model_path, map_location="cpu")

    print("=== TOP-LEVEL TYPE ===")
    print(type(obj))
    print()

    if isinstance(obj, dict):
        print("=== TOP-LEVEL KEYS ===")
        for k in obj.keys():
            print(k)
        print()
    else:
        print("Top-level object is not a dict; trying to treat it as a state_dict-like object.")
        print()

    state_dict = None
    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj.keys()) and any(hasattr(v, "shape") for v in obj.values()):
            state_dict = obj
        for candidate in ["state_dict", "model_state_dict", "model_state", "model"]:
            if candidate in obj and isinstance(obj[candidate], dict):
                state_dict = obj[candidate]
                break
    elif hasattr(obj, "keys"):
        state_dict = obj

    if state_dict is None:
        print("=== STATE_DICT ===")
        print("Unable to automatically identify a state_dict.")
        return

    print("=== STATE_DICT KEYS AND SHAPES ===")
    for k, v in state_dict.items():
        shape = tuple(v.shape) if hasattr(v, "shape") else type(v)
        print(f"{k}: {shape}")

    print()
    print("=== QUICK GUESSES ===")
    gru_keys = [k for k in state_dict.keys() if k.startswith("GRU_list.")]
    toout_keys = [k for k in state_dict.keys() if k.startswith("toOut.")]
    print(f"num_state_dict_entries = {len(state_dict)}")
    print(f"num_GRU_list_entries = {len(gru_keys)}")
    print(f"num_toOut_entries = {len(toout_keys)}")

    units = sorted({int(k.split('.')[1]) for k in gru_keys if k.split('.')[1].isdigit()})
    print(f"gru_unit_indices = {units}")

    hidden_channels = None
    for k, v in state_dict.items():
        if k.endswith("i2h.0.conv.weight") and hasattr(v, "shape") and len(v.shape) == 4:
            out_ch = v.shape[0]
            if out_ch % 3 == 0:
                hidden_channels = out_ch // 3
                break
    print(f"guessed_hidden_channels = {hidden_channels}")

    kernel_size = None
    for k, v in state_dict.items():
        if k.endswith("i2h.0.conv.weight") and hasattr(v, "shape") and len(v.shape) == 4:
            kernel_size = tuple(v.shape[-2:])
            break
    print(f"guessed_kernel_size = {kernel_size}")

    input_channels = None
    for k, v in state_dict.items():
        if k == "GRU_list.0.i2h.0.conv.weight" and hasattr(v, "shape") and len(v.shape) == 4:
            input_channels = v.shape[1]
            break
    print(f"guessed_first_cell_in_channels = {input_channels}")

    toout_out = None
    for k, v in state_dict.items():
        if k == "toOut.4.conv.weight" and hasattr(v, "shape") and len(v.shape) == 4:
            toout_out = v.shape[0]
            break
    print(f"guessed_toOut_last_out_channels = {toout_out}")


if __name__ == "__main__":
    main()