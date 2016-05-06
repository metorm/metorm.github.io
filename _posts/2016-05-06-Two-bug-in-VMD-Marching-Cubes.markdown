---
layout: post
title:  "记 VMD Marching Cubes 模块中的两个bug"
date:   2016-05-06 20:39:14 +0800
tags: [CUDA]
---

最近折腾 Level Set 算法，用到了[VMD（Visual Molecular Dynamics）库](http://www.ks.uiuc.edu/Research/vmd/)。这东西号称是美帝某核高基科研基金项目的产物，想来质量应该不错，关键是开源。岂料天下基金一般那啥，我不过是用到了其中的两个文件，也就是 Marching Cubes 算法的一个类，结果一个月之内踩到了两个坑。

# 幽灵崩溃：边界条件引发的惨案

某日改动一个地方，本来是与 Marching Cubes无关的，然而自那以后程序开始不定期崩溃，报内存错误，有点像是内存越界。关键是，TMD这个问题在我两台机器上表现的不一样。打开 CUDA 内存检查功能（自带程序减速20倍buff），笔记本（显卡GT 750M）显示 VMD 源码的一处有内存越界，然而台式机（显卡Quadro K2200）报错位置却是游移不定，唯一确定的是执行到某一行之后，任何分配显存的动作都会报错，注释掉一处就在下一处分配内存的地方保存。

当时一是对 VMD 的源码质量抱有幻想，而是觉得 Quadro K2200 这三千多一块的显卡，应该比笔记本靠谱，所以一直遵循中学老师教诲“从自身找原因”。然而山穷水尽之后，我开始分析 VMD 究竟干了些啥。

[Marching Cubes 算法](https://en.wikipedia.org/wiki/Marching_cubes)是很简单的，主要就是从距离场中抽取等值面。距离场中的每个格子是一个Cube，每个Cube中可能抽出0~4个三角形。具体有几个，数目是不固定的。然而，除了Tesla卡这种高富帅，GPU代码不能直接在GPU上申请内存。

VMD里面这一块，为了在GPU上面实现此算法，用到了前向求和操作。具体来说，就是先执行一次扫描，计算出每个Cube会抽出几个三角形，数据存入一个线性数组。然后前向求和：

    输入： 0 1 0 2 0 3 0 4
    输出： 0 0 1 1 3 3 6 6

结果中每一位等于输入中在这之前的数求和。有了这里的输出数据，就可以预先分配空间，每个Cube前面分配了多少个三角形的空间也是固定的，因此自己直接接着后面写入就可以，方法就我来看还是很巧妙。这算法有一个需要特别处理的地方，就是最后一位。前向求和是不处理最后一位的，所以必须手动处理。

令人抓跨的是， VMD 的作者在这里把正负号搞反了……

下面是我给作者的邮件：


>    I download a release version of VMD several months ago so I cannot recall
>    the exactly version tag. But on this web page
>    ([1]http://www.ks.uiuc.edu/Research/vmd/doxygen/CUDAMarchingCubes_8cu-source.html)
>    the problem still exists, at line 00313:
> 
>
>```C++
>00313 // compact voxel array
>00314 __global__ void
>00315 // __launch_bounds__ ( KERNTHREADS, 1 )
>00316 compactVoxels(unsigned int * RESTRICT compactedVoxelArray,
>00317               const uint2 * RESTRICT voxelOccupied,
>00318               unsigned int lastVoxel, unsigned int numVoxels,
>00319               unsigned int numVoxelsp1) {
>00320     unsigned int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;
>00321     unsigned int i = (blockId * blockDim.x) + threadIdx.x;
>00322
>00323     if ((i < numVoxels) && ((i < numVoxelsp1) ? voxelOccupied[i].y <
>voxelOccupied[i+1].y : lastVoxel)) {
>00324       compactedVoxelArray[ voxelOccupied[i].y ] = i;
>00325     }
>00326 }
>```
>
>
>    This kernel function is supposed to make a list of active voxels. I think
>    in line 323, the judgement conditions shall be like:
> 
>    
> ```
>    (i < numVoxels) && ((i < numVoxels - 1) ? voxelOccupied[i].y <
>    voxelOccupied[i + 1].y : lastVoxel)
> ```
>    
> 
>    The purpose of introducing numVoxelsp1 is to avoid reading from
>    voxelOccupied[numVoxels], which is out of the array. But when the function
>    is called, as numVoxelsp1 is set to numVoxels +1, meaning that when i<
>    numVoxels is true, i < numVoxelsp1 is always true, and the program would
>    never read from lastVoxel. Sometimes the last voxel will be judged as
>    active incorrectly, and the ```generateTriangleVerticesSMEM``` function called
>    after this will write to illegal location due to this.
> 
>    
> 
>    Under relatively large grid size, the program usually runs well. But in
>    the last few days I tried a grid size of [24 96 96], and the program would
>    refuse to allocate any global memory after calling computeIsosurfaceVerts
>    function. I find this problem and modified the kernel like below:
> 
>    
>```C++
>// compact voxel array
> __global__ void
> // __launch_bounds__ ( KERNTHREADS, 1 )
> compactVoxels(unsigned int * RESTRICT compactedVoxelArray,
>               const uint2 * RESTRICT voxelOccupied,
>               unsigned int lastVoxel, unsigned int numVoxels) {
>     unsigned int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;
>     unsigned int i = (blockId * blockDim.x) + threadIdx.x;
> 
>          if ((i < numVoxels) && ((i < numVoxels - 1) ? voxelOccupied[i].y
>< voxelOccupied[i + 1].y : lastVoxel)) {
>       compactedVoxelArray[ voxelOccupied[i].y ] = i;
>     }
>}
>```
>    
> 
>    And everything worked find after that.

后面就不多说了，作者确认了我的发现是正确的。


# 精度损失：就差那么一点

依然是与 Level Set 集成时候暴露出来的问题。流程中一个地方要做几何体的布尔运算。我是用与距离场相配合的一些数值技巧来处理的。细节不多说，关键是，在两个几何体很相近，但是大小不同，基本上是俄罗斯套娃那样一个套一个的情况下，这丫的给出的结果误差大的惊人。然而，一旦两个几何体的表面之间的间隙比较大，误差就很小了。

开始的时候，也没把这个当回事。数值方法嘛，也算可以理解，先凑合用。然而，这两天写论文（跑个题，初审审稿人鄙视我，切！），要解决这些遗留问题了，发现这问题不是一般的顽固，各种调整参数，加密网格，一无所获。这误差连减小的趋势都没有。

更可恨的是，网格越细，出问题的数据点越多，一反常态。极端情况下，上蹿下跳的数据直接污染了结果曲线中的前十个数据点（总共300个），连当野点除掉的借口都没有，这图线画在论文里，会被那丫的审稿人再鄙视一遍的。

仔细想想，这不该是几何体相切引起的偶然性误差了，就是系统误差。恰好当时复查代码到了纹理抓取这一段（参见下面PS）。纹理抓取方法我验证过了没问题，既然抓取纹理的坐标没问题，那么会不会是 Marching Cubes 本身取出来的曲面有偏移？大小两个俄罗斯套娃没有套正，发生了干涉，于是一大块曲面被切掉了，再统计面积自然变小了——恰恰，上面说的误差，也是偏小为主。

于是输出一些几何参数，发现 Level Set 场的位置和边界（BoundingBox）与 Marching Cubes 不符。就是它了，理论上这里应该是完全相符的。查那么几毫米正好是两个几何体的间隙那个数量级。顺藤摸瓜，发现 Marching Cubes 的初始化函数里面，计算每个 Cube 的尺寸时候，用的是：

    Cube边长=总长/节点数量

瞎胡搞嘛，幼儿园数学题，从零开始隔十米种一棵树，到一百米远处一共有几棵树？可见美帝中小学的数学教育确实有点那啥。修改源码为：

    Cube边长=总长/(节点数量-1)

天下太平。

# PS：CUDA 的纹理抓取

[来源](http://www.xuebuyuan.com/601346.html)

>最近在做个cuda的项目,其中利用到了纹理内存去加速数据读取和插值过程.因为有些细节没有充分注意到.直接导致这个项目的进展缓慢.数据精度问题很大.没有办法,只能是进行一步步地排查,到了今天才将这个问题彻底解决.而这一切的原因只是纹理索引使用上的一个不注意.OK闲话少说,直接给出注意事项.
>
>cuda中的纹理访问可以分为最近点插值cudaFilterModePoint和线性插值cudaFilterModeLinear.而这次项目的问题就是出在线性插值上.在某些通用计算中我们使用3Dtexture并不是因为整体的数据都具有相关性,可能这些数据仅仅具有2D的相关性,但是为了处理方便才把他们同时绑定到一起.然后利用循环读取各个2D数据的值在进行处理. 但是cuda在做线性插值前会先将索引值减去0.5,因此直接使用循环时的索引或者其他整数索引时就会导致计算错误,无法获得我们想要的值. 既然已经知道了原因一切就好办了,直接索引+0.5后再处理,然后就OK了.
>
>厄 要说的注意事项说完了,就是这么简单,可是没注意到的话你就只能天天郁闷着了.最近总是在犯一些奇奇怪怪的小问题,都是一些写程序时不注意的地方.如果只是处略的进行测试看不出来什么问题,但是问题叠加起来后就都显现出来了.看来 不论要做的项目有多简单,或者自己对他的实现有多了解,更或者这只是个算法实现而非工程类项目(我现在做的就是这个),都应该养成良好的开发习惯,并且对中间步骤进行详尽的单元测试.否则最后开发的时间会更长,问题也会更多.看来需要补一补软件工程方面的知识了.

