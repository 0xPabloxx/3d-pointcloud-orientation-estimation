# 点云对称性标注工具 - 使用指南

## ✨ 断点续标功能

### 🎯 核心特性

**完全支持断点续标！** 无论何时关闭，下次启动会自动继续。

---

## 📖 使用场景

### 场景1：正常标注流程

```bash
# 第一天 - 启动工具
python annotate_symmetry_web_v2.py

# 在浏览器中标注（http://localhost:8051）
# 标注了 50 个样本
# 工作结束，关闭浏览器
# Ctrl+C 停止工具
# 关机回家
```

**✅ 标注会自动保存到**：
- `data/symmetry_annotations.json`
- `data/symmetry_annotations.csv`
- `data/symmetry_annotations.md`

---

### 场景2：第二天继续标注

```bash
# 第二天 - 重新启动工具
python annotate_symmetry_web_v2.py

# 输出会显示：
# [Annotation] Loaded 50 existing annotations
# [Resume] Starting from sample 51/12311: airplane_0051.ply
# ✅ 自动从第51个样本开始！
```

**在浏览器中打开**，你会看到：
- 📊 进度显示：50/12311 (0.4%)
- 🎯 自动定位到第一个未标注的样本
- ✅ 所有已标注的样本都被保留

---

## 🔐 数据安全保障

### 1. **实时自动保存**
每次点击"对称类型"+"正面方向"后立即保存：
```
✅ 已保存！对称性: 1个正面, 方向: -Z
```

### 2. **三重备份**
同时保存三种格式：
- JSON（完整数据）
- CSV（表格）
- Markdown（报告）

### 3. **崩溃恢复**
即使程序意外关闭：
- ✅ 已标注的数据不会丢失
- ✅ 重新启动自动恢复
- ✅ 从最后一个标注继续

---

## 💡 实用技巧

### 技巧1：随时查看进度

```bash
# 查看已标注数量
cat data/symmetry_annotations.json | grep "file" | wc -l

# 查看Markdown报告
cat data/symmetry_annotations.md
```

### 技巧2：批量标注策略

**建议工作流**：
1. 每次标注 100-200 个样本
2. 休息 10 分钟（避免疲劳）
3. 查看报告，检查质量
4. 继续标注

### 技巧3：按类别标注

如果想先标注特定类别（如glass_box）：
```bash
# 只标注glass_box
python annotate_symmetry_web_v2.py \
  --data_dir data/full_mn40_normal_resampled_ply/glass_box \
  --output data/annotations_glassbox.json
```

---

## 🛡️ 数据完整性检查

### 检查标注是否完整

```python
import json

with open('data/symmetry_annotations.json', 'r') as f:
    annotations = json.load(f)

# 检查不完整的标注
incomplete = []
for file_path, ann in annotations.items():
    if not ann.get('front_direction'):
        incomplete.append(file_path)

if incomplete:
    print(f"⚠️ 发现 {len(incomplete)} 个不完整的标注")
else:
    print(f"✅ 所有 {len(annotations)} 个标注都完整")
```

---

## 📊 进度追踪

### 方式1：查看Markdown报告
```bash
cat data/symmetry_annotations.md
```

输出示例：
```markdown
## 📊 标注统计

- **总标注数**: 50/12311
- **完成进度**: 0.4%

### 对称性分布
| 对称类型 | 数量 | 占比 |
|---------|------|------|
| 1个正面 | 25 | 50.0% |
| 4个正面 | 15 | 30.0% |
...
```

### 方式2：使用处理工具
```bash
python process_annotations.py --mode stats
```

---

## ⚙️ 高级配置

### 更改输出位置
```bash
python annotate_symmetry_web_v2.py \
  --output /path/to/my_annotations.json
```

### 更改端口
```bash
python annotate_symmetry_web_v2.py --port 9999
# 浏览器访问: http://localhost:9999
```

### 只标注特定类别
```bash
# 只标注chair和table
mkdir -p data/selected_categories
cp -r data/full_mn40_normal_resampled_ply/chair data/selected_categories/
cp -r data/full_mn40_normal_resampled_ply/table data/selected_categories/

python annotate_symmetry_web_v2.py \
  --data_dir data/selected_categories \
  --output data/annotations_chair_table.json
```

---

## 🔧 故障排除

### 问题1：标注没有保存？
**检查**：
- 是否选择了"对称类型" + "正面方向"两个？
- 查看右侧面板是否显示"✅ 已保存"？

### 问题2：重启后索引不对？
**原因**：可能JSON文件损坏
**解决**：
```bash
# 备份当前标注
cp data/symmetry_annotations.json data/symmetry_annotations_backup.json

# 手动修复或重新开始
```

### 问题3：找不到某个已标注的样本？
**解决**：
```bash
# 在JSON中搜索
cat data/symmetry_annotations.json | grep "airplane_0001"

# 或使用处理工具分类查看
python process_annotations.py --mode classify
```

---

## 📝 标注质量检查清单

### 每次标注后检查：
- [ ] 标注数量是否增加？
- [ ] CSV文件是否更新？
- [ ] Markdown报告是否正确？

### 定期检查（每100个样本）：
- [ ] 对称性分布是否合理？
- [ ] 矫正率是否符合预期？
- [ ] 有无异常数据？

---

## 🎯 总结

**断点续标的三大优势**：

1. **安全可靠**
   - 实时自动保存
   - 三重备份格式
   - 崩溃恢复

2. **灵活高效**
   - 随时暂停
   - 自动续标
   - 进度透明

3. **易于管理**
   - 多种查看方式
   - 完整性检查
   - 质量追踪

**你可以放心地**：
- ✅ 随时关机回家
- ✅ 分多天完成标注
- ✅ 中途修改已标注数据
- ✅ 在不同电脑上继续（只需复制JSON文件）

---

**祝标注顺利！🎉**
