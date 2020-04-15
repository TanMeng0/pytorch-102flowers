import os
from shutil import copyfile     #copyfile(原路径，新路径)   拷贝文件，路径必须是完整的文件路径

path1 = "./oxford-102-flowers/train"
path2 = "./oxford-102-flowers/test"
path3 = "./oxford-102-flowers/valid"

dataTxt1 = "./oxford-102-flowers/train.txt"
dataTxt2 = "./oxford-102-flowers/test.txt"
dataTxt3 = "./oxford-102-flowers/valid.txt"

#原数据是一个jpg文件夹和train.txt test.txt valid.txt三个文件，这里根据三个txt文件内容将jpg文件夹拆分为train test valid三个文件夹,每个文件夹内又根据不同label，一个label生成一个文件夹。
def SpliteDataset(path, dataTxt):
    if not os.path.exists(path):
        os.mkdir(path)
    with open(dataTxt, 'r') as f:
        pathList = f.readlines()
        print(pathList)
        for i in range(0, len(pathList)):
            pathTemp = pathList[i].split()[0]          #获取一行中的路径；   一行数据的原格式为“路径 label”
            labelTemp = pathList[i].split()[1]
            filename = pathTemp.split('/')[-1]         #获取路径中的文件名
            if not os.path.exists(path+'/'+labelTemp):
                os.mkdir(path+'/'+labelTemp)
            if not os.path.exists(path+'/'+labelTemp+'/'+filename):
                copyfile('./oxford-102-flowers/'+pathTemp, path+'/'+labelTemp+'/'+filename)


if __name__ == "__main__":
    SpliteDataset(path1, dataTxt1)
    SpliteDataset(path2, dataTxt2)
    SpliteDataset(path3, dataTxt3)
