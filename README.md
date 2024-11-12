# imgclass
python 的图形分类封装包


## 1. 数据预处理
- prepro.py 
- 返回torch.utils.data.DataLoader
```
from prepro import run_pre
dataloader = run_pre(data_path, stage="train", batch_size=8, normalize=False)
```
## 2. 训练模型



## 3. 测试模型