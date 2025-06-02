with open("/home/tc415/muPPIt/dataset/temp.txt", 'r') as f:
    lines = f.readlines()

new_lines = ["DUSP12,MGNGMNKILPGLYIGNFKDARDAEQLSKNKVTHILSVHDSARPMLEGVKYLCIPAADSPSQNLTRHFKESIKFIHECRLRGESCLVHCLAGVSRSVTLVIAYIMTVTDFGWEDALHTVRAGRSCANPNVGFQRQLQEFEKHEVHQYRQWLKEEYGESPLQDAEEAKNILAAPGILKFWAFLRRL:" + line for line in lines]

with open("/home/tc415/muPPIt/dataset/phosphatases.csv", 'a') as f:
    for line in new_lines:
        f.write(line)
    f.write('\n')