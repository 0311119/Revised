# verify_npy.py
import numpy as np
from pathlib import Path

# 修改为你的实际路径
eeg_path = Path(r"c:\Desktop\Milmer\code\EEGData\s01_eeg.npy")
lab_path = Path(r"c:\Desktop\Milmer\code\EEGData\s01_labels.npy")

def safe_load_info(path):
    arr = np.load(path, mmap_mode="r")
    return arr.shape, arr.dtype

def main():
    eeg_shape, eeg_dtype = safe_load_info(eeg_path)
    lab_shape, lab_dtype = safe_load_info(lab_path)

    print("EEG file:", eeg_path)
    print("  shape:", eeg_shape)
    print("  dtype:", eeg_dtype)
    print("LABEL file:", lab_path)
    print("  shape:", lab_shape)
    print("  dtype:", lab_dtype)

    # 基本一致性检查
    ok_trials = (len(eeg_shape) >= 1 and eeg_shape[0] == 800) and (len(lab_shape) == 1 and lab_shape[0] == 800)
    print("\n是否按 trial 划分（首维=800 且标签数=800）:", "YES" if ok_trials else "NO")

    if len(eeg_shape) >= 2:
        print("EEG 通道数（第2维）:", eeg_shape[1])
    if len(eeg_shape) >= 3:
        print("EEG 采样点（第3维）:", eeg_shape[2])

if __name__ == "__main__":
    main()