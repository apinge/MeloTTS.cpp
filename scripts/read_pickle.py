import pickle

# 定义读取字典并保存为 UTF-8 文本文件的函数
def save_dict_to_txt(pickle_file, txt_file):
    try:
        # 从 pickle 文件中加载数据
        with open(pickle_file, 'rb') as f:
            cmudict_cache = pickle.load(f)
        
        # 检查数据是否为字典
        if isinstance(cmudict_cache, dict):
            # 打开目标文本文件，并指定 utf-8 编码
            with open(txt_file, 'w', encoding='utf-8') as txt_f:
                # 写入字典内容到文本文件
                for key, value in cmudict_cache.items():
                    #txt_f.write(f"{key}: {value}\n")
                     # 仅当键不以“(2)”结尾时才写入文件
                    if not key.endswith('(2)'):
                        # 将列表转换为集合格式的字符串并转换为小写
                        value_str = ','.join(f"{' '.join(map(str.lower, v))}" for v in value)
                        txt_f.write(f"{str(key).lower()}:{str(value_str)},\n")
            print(f"字典已保存到 {txt_file} (UTF-8 编码)")
        else:
            print("加载的对象不是字典。")
    
    except FileNotFoundError:
        print(f"文件 '{pickle_file}' 未找到。")
    except Exception as e:
        print(f"读取文件时出错: {e}")

# 调用函数，传入pickle文件名和目标txt文件名
save_dict_to_txt('cmudict_cache.pickle', 'cmudict_cache.txt')
