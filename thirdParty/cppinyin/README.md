# 简介

cppinyin 是一个用 C++ 实现的中文转拼音工具，可作为[pypinyin](https://github.com/mozillazg/python-pinyin) 的替代方案，之所以使用 C++ 实现，一是方便部署，二也是为了提升效率，我相信 cppinyin 的转换速度要比 pypinyin 好不少。


# 安装

## python

```
pip install cppinyin
```

## C++

可从源码编译:

```
mkdir build
cd build
cmake ..
make -j
```

集成可用 cmake


# 使用

## python

```
import cppinyin

encoder = cppinyin.Encoder()

encoder.encode("我是中国人民的儿子")
```

## 命令行

```
cppinyin encode 我是中国人民的儿子
```

