"""Print the device this course will train on. CPU is a pass."""

import torch


def main() -> None:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / (1024**3)
        print(f"PASS cuda  {name}  {vram:.1f} GB  dtype float32")
    else:
        print("PASS cpu  dtype float32  (install the cu118 torch wheel to use the GPU)")


if __name__ == "__main__":
    main()
