# AEGIS Demo v0.2 Cloud Demo

这是一个可部署到 Streamlit Community Cloud 的轻量演示版。

本 demo 使用 WTBD 数据集的一小部分真实样本，展示如何对 MLLM 结构化缺陷诊断进行属性级可靠性验证，并展示未来 AEGIS-Net 的模型化升级路线。

## 功能

### 页面 1：AEGIS 属性级验证原型

- 显示真实风电叶片缺陷图像。
- 读取 PASCAL VOC XML 标注。
- 支持一图多目标 bbox 选择。
- 显示 bbox 和缺陷 patch。
- 根据 XML 类别验证 MLLM 诊断中的“类型”属性。
- 根据 bbox 中心点九宫格位置验证“位置”属性。
- 将“严重度”和“形态”明确标记为暂未启用。
- 输出属性级验证表和自动复核反馈。

### 页面 2：AEGIS-Net 训练与推理展示

- 展示当前规则原型与未来 AEGIS-Net 的关系。
- 展示 MVP 数据集格式、jsonl 单行样本、batch 输入形状和结构化输出分数。
- 展示静态学术风格模型结构图：输入层、编码层、融合层、属性 query 层、cross-attention 层、证据层、预测头和输出层。
- 展示训练过程：属性一致性标签、label mask、损失函数和动态输出示意。
- 展示推理过程：正式推理只输入图像、bbox、patch 和 MLLM 诊断 JSON，不输入 XML 真值。

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

1. 将本文件夹内容作为 GitHub 仓库内容上传。
2. 登录 Streamlit Community Cloud。
3. 新建 app，选择该 GitHub 仓库。
4. Main file path 填写：

```text
app.py
```

5. 点击 Deploy。

部署成功后，可以把生成的 `*.streamlit.app` 链接发给其他电脑访问。

## 原型边界

当前版本不是已经训练完成的 AEGIS-Net。

当前版本中：

- 类型验证使用 XML 专家标注作为参考。
- 位置验证使用 bbox 几何计算作为参考。
- 严重度和形态因为缺少属性级真值，暂不自动验证。
- 页面 2 的模型结构、训练和推理分数是研究设计展示，不作为最终模型性能结果。

正式 AEGIS-Net 后续应通过视觉编码器、文本编码器、bbox 几何编码器、cross-attention 和属性 query 学习图像证据与诊断文本属性之间的一致性。
