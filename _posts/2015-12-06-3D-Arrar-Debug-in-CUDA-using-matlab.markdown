---
layout: post
title:  "科学计算编程中3D数组的调试技巧"
date:   2015-12-06 20:00:00 +0800
tags: [CUDA,Matlab]
---

涉及科学计算或者仿真的程序的编写，总是绕不开大规模的数组，其中比较常见又折磨人的大约就是三维数组了，动辄512x512x1024的数组，各种微分算法下各种下标来回倒腾，某些情况下（例如CUDA）还被迫用一维数组来代替，要自己负责坐标转换。万一出错（其实不出错才是万一）几乎没法调试，只知道结果很奇怪，但是即使断点进去，也只是看到一个个独立的数字，很多时候相邻的元素都不在一起。这个时候，写C/C++的人就会无比怀念MATLAB——可以任意时候对一个数组的一个剖面surf()一下看看趋势。
# 其实是有办法很方便的将C++的数组搞进MATLAB画图的
材料：
* MATLAB 2015 稍微老一点的版本应该也行
* Excel 2016 老版本不一定行
* 任意文本编辑器
* 你的源码


## 写一个输出函数
第一步要把3D数组数据转移到磁盘文件，这一步最慢，但是磨刀不误砍柴工。放一个我调试CUDA的例子：


```
void   DebugPrintThrustDevice3DArrayAlong ( thrust :: device_vector < real > & devVector ,
     uint3   meshdim ,  DebugOutputInfo  & DebugInfo ,  std :: string   ArrayName )
{
     if  (! DebugInfo . DebugOutput )  return ;
     DebugInfo . FileOutputStream  <<  std :: setprecision (4); // << std::setiosflags(std::ios::scientific);
     DebugInfo . FileOutputStream  <<  std :: endl  <<  "#"  <<  ArrayName  <<  ":"  <<  std :: endl ;
     thrust :: host_vector < real >  Value ( devVector );

     for  ( size_t   x  = 0;  x  <  meshdim . x ;  x  +=  DebugInfo . PrintGap )
    {
         for  ( size_t   y  = 0;  y  <  meshdim . y ;  y  +=  DebugInfo . PrintGap )
        {
             for  ( size_t   z  = 0;  z  <  meshdim . z ;  z  +=  DebugInfo . PrintGap )
            {
                 if  ( DebugInfo . OutputToFile )
                {
                     DebugInfo . FileOutputStream  <<  std :: setfill ( ' ' ) <<  std :: setw (10) <<  Value [ Axyz2idx ( make_uint3 ( x ,  y ,  z ),  meshdim )];
                }
                 else
                {
                     std :: cout  <<  std :: setw (6) <<  Value [ Axyz2idx ( make_uint3 ( x ,  y ,  z ),  meshdim )] <<  " " ;
                }
            }
             if  ( DebugInfo . OutputToFile )
            {
                 DebugInfo . FileOutputStream  <<  std :: endl ;
            }
             else
            {
                 std :: cout  <<  std :: endl ;
            }
        }
         if  ( DebugInfo . OutputToFile )
        {
             DebugInfo . FileOutputStream  <<  std :: endl  <<  std :: endl ;
        }
         else
        {
             std :: cout  <<  std :: endl  <<  std :: endl ;
        }
    }
     std :: cout  <<  "# Array "  <<  ArrayName  <<  "debug output finished. #"  <<  std :: endl ;
}
```

这是一个 CUDA Thrust 3D 数组的输出函数，基本操作跟 vector 一样。*DebugInfo . FileOutputStream* 里面已经封装好了一个打开的文件。
如果有些奇怪的数组，自己用奇怪的方式转换。上面的函数输出格式是一个切片一个切片地输出。

## 转成excel数据
这一步主要是利用Excel对文本的处理能力，用文本编辑器打开，复制你想要查看的数据，到Excel中，粘贴——按Ctrl键，这时候会弹出“文本导入向导”，就跟导入txt文件一样，只不过数据源换成了剪贴板。选择固定宽度，或者上面输出函数里你用了逗号什么的也可以用分隔符。我选择固定宽度主要是文本好看。
在Excel 2016之前没注意这个“文本导入向导”是否出现在“选择性粘贴”之中。起码2003肯定是没有的……
## 导入MATLAB，绘图
接下来就简单了，复制，到MATLAB里面随意新建一个变量，双击，会出现跟Excel类似的表格编辑器，贴进去，surf()一下，啥都看清楚了。
表格式编辑器好像是在MATLAB 2012左右的时候引入的，再老的版本应该不行。

## 高手还可以写个自动导入脚本
写完记得发给我:heart: