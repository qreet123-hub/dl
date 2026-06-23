# 课设文字输出程序
# 此程序仅用于在控制台输出“我的课设”四个字

def main():
    # 定义要输出的文字
    text = "我的课设"
    
    # 使用print函数输出文字
    print(text)
    
    # 可选：增加一些装饰性输出，使显示更醒目
    print("-" * 20)
    print(f"【{text}】")

# 程序入口
if __name__ == "__main__":
    main()