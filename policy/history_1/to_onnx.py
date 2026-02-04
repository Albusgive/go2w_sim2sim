import torch

# 加载 LibTorch 使用的同一个 .pt 文件
model = torch.jit.load("policy.pt")  # 无需 C++ 环境！
model.eval()

# 根据实际输入构造 dummy_input（53维特征向量）
dummy_input = torch.randn(1, 53)  # 或 torch.randn(53)（无batch维度）

# 尝试导出（注意：成功率取决于模型结构）
try:
    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=13,
        # 如遇自定义算子报错，可尝试（但ONNX Runtime可能不支持）：
        # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    )
    print("✅ ONNX 导出成功！")
except Exception as e:
    print(f"❌ 导出失败: {type(e).__name__}: {e}")
    print("💡 建议：1. 检查输入形状 2. 联系模型提供方获取原始PyTorch源码")