# -*- coding:utf-8 -*-
from numpy import *
from tkinter import *
import regressionTrees
import matplotlib

matplotlib.use("TkAgg")  # 设定后端为TkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def reDraw(tolS, tolN):
    reDraw.f.clf()  # 清空之前的图像
    reDraw.a = reDraw.f.add_subplot(111)  # 重新添加子图
    if chkBtnVar.get():  # 检查复选框是否选中，确定是模型树还是回归树
        if tolN < 2:
            tolN = 2
        myTree = regressionTrees.createTree(reDraw.rawDat, regressionTrees.modelLeaf, regressionTrees.modelErr,
                                            (tolS, tolN))
        yHat = regressionTrees.createForeCast(myTree, reDraw.testDat, regressionTrees.modelTreeEval)
    else:  # 回归树
        myTree = regressionTrees.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = regressionTrees.createForeCast(myTree, reDraw.testDat)
    reDraw.a.scatter(array(reDraw.rawDat[:, 0]), array(reDraw.rawDat[:, 1]), s=5)  # 画真实值的散点图
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)  # 画预测值的直线图
    reDraw.canvas.draw()


def getInputs():  # 获取用户输入的值，tolN期望得到整数值，tolS期望得到浮点数，
    try:
        tolN = int(tolNentry.get())  # 在Entry部件调用get方法，
    except:
        tolN = 10
        print("输入int型数值作为tolN")
        tolNentry.delete(0, END)
        tolNentry.insert(0, "10")
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("输入浮点型数值作为tolS")
        tolSentry.delete(0, END)
        tolSentry.insert(0, "1.0")
    return tolN, tolS


def drawNewTree():  # 有人点击ReDraw按钮时就会调用该函数
    tolN, tolS = getInputs()  # 得到输入框的值
    reDraw(tolS, tolN)


root = Tk()
# Label(root, text="绘制占位符").grid(row=0, columnspan=3)  # 设置文本，第0行，距0的行值为3,
reDraw.f = Figure(figsize=(5, 4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.draw()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

Label(root, text="tolN").grid(row=1, column=0)
tolNentry = Entry(root)  # Entry为允许单行文本输入的文本框，设置文本框，再定位置第1行第1列，再插入数值
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, "10")

Label(root, text="tolS").grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, "1.0")

Button(root, text="重画" ,command=drawNewTree).grid(row=1, column=2, rowspan=3)  # Botton按钮，设置第1行第2列，列值为3
chkBtnVar = IntVar()  # IntVar为按钮整数值小部件
chkBtn = Checkbutton(root, text="ModelTree",variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

reDraw.rawDat = mat(regressionTrees.loadDataSet("data/sine.txt"))
reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
reDraw(1.0, 10)
root.mainloop()




