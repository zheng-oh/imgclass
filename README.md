# imgclass
python 的图形分类封装包
# 训练代码
```python
import sys
sys.path.append("/Users/xingzheng/Documents/pypkgs/imgclass")
from training import Train

def run():
    name = "duck_200_ie"
    tr = Train(name,"./data/IE",18,num_epochs=60)
    tr.run()
    print("结束")

if __name__ == '__main__':
    run()
```
