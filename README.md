# 动作生成器

基于 [MoMask](https://github.com/EricGuo5513/momask-codes) 的动作生成系统，提供了简单的 Web 界面来生成动作。

## 目录结构

```
.
├── api_server.py      # API 服务器
├── momask_test.py     # Gradio Web 界面
├── base_option.py     # 配置文件
└── generation/        # 生成的动作文件存放目录
    └── default/       # 默认输出目录
        ├── animations/  # 动画文件
        └── joints/     # 关节数据
```

## 安装步骤

1. 首先确保已经安装了 MoMask 项目的依赖

2. 安装额外依赖
```bash
# 在 MoMask 项目目录下安装 API 服务器依赖
pip install fastapi uvicorn

# 在运行 Web 界面的目录下安装 Gradio
pip install gradio
```

3. 替换文件
   - 将 `base_option.py` 复制到 MoMask 项目的 `options` 目录下并替换
   - 将 `api_server.py` 复制到 MoMask 项目根目录
   - 将 `momask_test.py` 放在要运行 Web 界面的目录下

## 运行说明

1. 启动 API 服务器
```bash
# 在 MoMask 项目目录下运行
python api_server.py
```
服务器启动后：
- API 接口地址：http://localhost:8000/generate_motion
- API 文档地址：http://localhost:8000/docs

2. 启动 Web 界面
```bash
# 在放置 momask_test.py 的目录下运行
python momask_test.py
```
Web 界面访问地址：http://localhost:7860

## 使用说明

1. Web 界面参数说明：
   - GPU ID：使用的 GPU 编号（-1 表示使用 CPU）
   - 输出文件夹名称：生成文件的存放目录名（默认为 default）
   - 输入动作描述文本：直接输入英文描述文本
   - 或上传文本文件：上传包含动作描述的 txt 文件

2. 生成的文件说明：
   - 动画文件：保存在 `generation/{输出文件夹名称}/animations/0/` 目录下
   - 关节数据：保存在 `generation/{输出文件夹名称}/joints/0/` 目录下
   - 生成的视频包含普通版本和带 IK 的版本（带 _ik 后缀）





