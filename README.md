# Kaggle竞赛代码仓库

这是我的Kaggle竞赛项目仓库，用于存放各种机器学习竞赛的代码和解决方案。

## 目录结构

```
├── kaggle-competitions/    # 各个Kaggle竞赛的项目文件夹
├── data/                  # 数据集存放目录
├── notebooks/             # Jupyter笔记本文件
├── scripts/               # 工具脚本
├── utils/                 # 通用工具函数
└── README.md              # 项目说明
```

## 环境设置

建议使用Python虚拟环境来管理依赖包：

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## 快速开始

1. 克隆仓库到本地
2. 在iTerm中输入 `code` 快速切换到此目录
3. 为每个新的Kaggle竞赛在 `kaggle-competitions/` 目录下创建新文件夹
4. 将数据集放在 `data/` 目录下
5. 在 `notebooks/` 目录下进行数据探索和模型开发

## 提交代码

```bash
git add .
git commit -m "描述你的更改"
git push origin main
```