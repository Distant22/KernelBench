import torch

def check_environment():
    print("=== CUDA 檢查 ===")
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch 是否能使用 CUDA: {cuda_available}")
    
    if cuda_available:
        print(f"當前 GPU 裝置: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
        # 測試建立一個 GPU 張量
        x = torch.rand(3, 3).cuda()
        print("成功在 GPU 上建立 Tensor!")
    else:
        print("❌ 找不到 CUDA 環境，請確認 NVIDIA 驅動與 Toolkit 是否安裝正確。")

    print("\n=== Triton 檢查 ===")
    try:
        import triton
        import triton.language as tl
        print(f"Triton 版本: {triton.__version__}")
        print("成功匯入 Triton 模組!")
    except ImportError:
        print("❌ 找不到 Triton，請確認是否成功安裝。")

if __name__ == "__main__":
    check_environment()