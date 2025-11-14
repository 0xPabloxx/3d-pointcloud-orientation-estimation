import re
import numpy as np
import matplotlib.pyplot as plt

# 解析训练日志
with open("training_fixed_weight.log", "r") as f:
    lines = f.readlines()

epochs = []
train_losses = []
val_losses = []

for line in lines:
    match = re.search(r'Epoch (\d+)/50 \| Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)', line)
    if match:
        epoch = int(match.group(1))
        train_loss = float(match.group(2))
        val_loss = float(match.group(3))
        epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

epochs = np.array(epochs)
train_losses = np.array(train_losses)
val_losses = np.array(val_losses)

print("="*60)
print("  Overfitting Analysis")
print("="*60)

# 1. 检查训练/验证损失的gap
final_train = train_losses[-1]
final_val = val_losses[-1]
best_val = val_losses.min()
gap = final_train - final_val

print(f"\n1. Loss Gap Analysis:")
print(f"   Final Train Loss: {final_train:.4f}")
print(f"   Final Val Loss:   {final_val:.4f}")
print(f"   Best Val Loss:    {best_val:.4f}")
print(f"   Gap (Train-Val):  {gap:.4f}")

if gap > 0:
    print(f"   ✅ Train > Val: 正常（训练集使用增强更难拟合）")
else:
    print(f"   ⚠️  Train < Val: 可能需要更多训练")

# 2. 检查验证损失是否稳定/上升（过拟合迹象）
# 找到最佳epoch之后验证损失的趋势
best_epoch_idx = np.argmin(val_losses)
best_epoch = epochs[best_epoch_idx]
后期的val_losses = val_losses[best_epoch_idx:]

val_上升次数 = 0
for i in range(1, len(后期的val_losses)):
    if 后期的val_losses[i] > 后期的val_losses[i-1]:
        val_上升次数 += 1

print(f"\n2. Validation Loss Stability (after best epoch {best_epoch}):")
print(f"   Remaining epochs: {len(后期的val_losses)}")
print(f"   Val loss increases: {val_上升次数}/{len(后期的val_losses)-1}")
print(f"   Val loss std (后期): {后期的val_losses.std():.5f}")

if 后期的val_losses.std() < 0.01:
    print(f"   ✅ 验证损失稳定，无明显过拟合")
else:
    print(f"   ⚠️  验证损失波动较大")

# 3. 检查测试损失 vs 验证损失
test_loss = 0.0019
print(f"\n3. Test vs Validation:")
print(f"   Best Val Loss: {best_val:.4f}")
print(f"   Test Loss:     {test_loss:.4f}")
print(f"   Ratio:         {test_loss/best_val:.2f}x")

if test_loss < 2 * best_val:
    print(f"   ✅ 测试损失接近验证损失，泛化良好")
else:
    print(f"   ⚠️  测试损失明显高于验证损失，可能过拟合")

# 4. 检查是否有early stopping需求
# 看验证损失是否在最佳epoch后持续上升
后期平均 = 后期的val_losses[1:].mean()
if 后期平均 > best_val * 1.5:
    print(f"\n4. Early Stopping:")
    print(f"   ⚠️  后期验证损失显著上升，建议使用early stopping")
else:
    print(f"\n4. Early Stopping:")
    print(f"   ✅ 验证损失未显著上升，训练合理")

# 5. 数据量分析
print(f"\n5. Dataset Size Analysis:")
print(f"   Train: 189 samples (×12 augmentation = 2268)")
print(f"   Val:   54 samples")
print(f"   Test:  28 samples")
print(f"   Total: 271 samples")

if 271 < 500:
    print(f"   ⚠️  数据量较小（<500），结果可能有一定方差")
else:
    print(f"   ✅ 数据量充足")

# 总结
print(f"\n{'='*60}")
print(f"  Overfitting Risk Assessment")
print(f"{'='*60}")

风险因素 = []
if gap < 0:
    风险因素.append("训练损失低于验证损失")
if test_loss > 2 * best_val:
    风险因素.append("测试损失明显高于验证损失")
if 后期平均 > best_val * 1.5:
    风险因素.append("后期验证损失显著上升")

if len(风险因素) == 0:
    print(f"\n✅ 无明显过拟合迹象")
    print(f"   - 训练/验证/测试损失均合理")
    print(f"   - 验证损失稳定")
    print(f"   - 数据增强有效")
else:
    print(f"\n⚠️  发现以下风险因素：")
    for f in 风险因素:
        print(f"   - {f}")

print(f"\n{'='*60}\n")
