from setuptools import setup, find_packages

setup(
    name="imgclass",  # 包名称
    version="0.1.4",  # 包版本
    author="ZXING",  # 作者名
    author_email="zxing_oh@163.com",  # 作者邮箱
    description="add 特征图可视化",  # 包描述
    long_description=open("README.md").read(),  # 从 README 加载长描述
    long_description_content_type="text/markdown",  # 长描述格式
    url="https://github.com/zheng-oh/imgclass",  # 项目地址
    packages=find_packages(),  # 自动找到子包
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",  # Python 版本要求
    install_requires=[
        "matplotlib>=3.9.3",
        "numpy>=2.1.3",
        "scikit_learn>=1.5.2",
        "setuptools>=75.1.0",
        "torch>=2.5.1",
        "torchvision>=0.20.1",
    ],
)
