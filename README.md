# AEGIS-Prototype v0.1 Cloud Demo

这是一个可部署到 Streamlit Community Cloud 的轻量演示版。

本 demo 使用 WTBD 数据集的一小部分真实样本，展示如何对 MLLM 结构化缺陷诊断进行属性级可靠性验证。

## 功能

- 显示真实风电叶片缺陷图像。
- 读取 PASCAL VOC XML 标注。
- 支持一图多目标 bbox 选择。
- 显示 bbox 和缺陷 patch。
- 根据 XML 类别验证 MLLM 诊断中的“类型”属性。
- 根据 bbox 中心点九宫格位置验证“位置”属性。
- 将“严重度”和“形态”明确标记为暂未启用。
- 输出属性级验证表和自动复核反馈。

## 数据

云端版内置小样本数据：

```text
sample_data/
├── JPEGImages/
├── Annotations/
├── class_definitions.txt
└── train_val_test_split.txt
```

这不是完整 WTBD 数据集，只用于在线演示。完整数据集请参考原始 Figshare 数据仓库。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud 部署

1. 将 `aegis_demo_cloud/` 作为 GitHub 仓库内容上传。
2. 登录 Streamlit Community Cloud。
3. 新建 app，选择该 GitHub 仓库。
4. Main file path 填写：

```text
app.py
```

5. 点击 Deploy。

部署成功后，可以把生成的 `*.streamlit.app` 链接发给其他电脑访问。

## 原型边界

当前版本不是训练好的 AEGIS-Net。

当前版本中：

- 类型验证使用 XML 专家标注作为参考。
- 位置验证使用 bbox 几何计算作为参考。
- 严重度和形态因为缺少属性级真值，暂不自动验证。

正式 AEGIS-Net 后续应通过视觉编码器、文本编码器和属性 query 学习图像证据与诊断文本属性之间的一致性。
