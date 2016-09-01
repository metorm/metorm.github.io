---
layout: post
title:  "CUDA矩阵乘法优化分析"
date:   2016-09-01 15:26:19 +0800
tags: [CUDA]
---

# 内容提要

+ 网上常见的 TILE 分块计算方法
+ 风辰《科学计算与企业级应用并行优化》中的优化方案

# Baseline

现有两个矩阵`A`、`B`，矩阵的宽度和高度表示为`A.width`, `A.height`等，`A.width = B.height`, 需要计算`A*B=C`.

矩阵乘法的基本公式不再重复。不考虑GPU这一特殊环境时，基本算法如下图：

![基础算法](/img/in-post/2016-09-01-CUDA-matrix-multiply-optimization/naive.png)

为了计算矩阵`C`中的一个元素，需要加载A中的一行和B中的一列。显然，如果按照最朴素的方式，在计算`C`中每个元素时直接加载相应的行、列，则`A`中每个元素将被加载`B.width`次，`B`中每个元素将被加载`B.height`次。该算法延迟很大，在GPU上甚至可能慢于CPU计算。

# TILE式分块计算

上述算法的主要问题是重复的读取操作过多。GPU编程中解决这一问题的常用方法是使用`__shared__`储存器，将需要的数据预加载入共享储存器，需要时再读取。然而，共享储存器大小有限，不可能将所有数据全部加载进共享储存器，只能是临时加载当前线程`block`需要的部分。

使用此思路，网上常见的一个矩阵乘法优化方案是：每个`block`负责计算`C`矩阵中一块`BS x BS`大小的区，`block`中每个线程负责计算该区域中的一个元素。这样，该`block`需要加载的数据局限在`A`矩阵中`BS x A.width`的一块，以及`B`矩阵中`B.height x BS`的一块，如下图：

![TILE 算法](/img/in-post/2016-09-01-CUDA-matrix-multiply-optimization/square-tiling.png)

加载数据时，每一个线程只从`A`、`B`中各加载一个对应于本线程在`block`中的位置的数据，并写入`__shared__`储存。这种算法下，`A`中每个元素将被加载`B.width/BS`次，`B`中每个元素将被加载`B.height/BS`次。`__global__`储存的负担直接减小了`BS`倍。

理论上，如果`__shared__`资源无限，`BS`自然是越大越好。`BS`大到极点时，整个矩阵都被加载进去了，每个元素只从`__global__`中读一次，就是前面说的“所有数据全部加载进共享储存器”了。

随后就是按矩阵相乘的做法，将加载进来的小矩阵进行乘加。由于此时数据都是从共享储存器中读取，速度比每次都到`__global__`中读取快得多。计算完上图中`BS x BS`大小的第一块后，将`A`中取元素的窗口向右“滑动”，将`B`中的窗口向下“滑动”，重复上述计算过程，将结果累加到上一次计算的结果中。至滑动完毕时，累加器中的数据即为对应位置的结果。这实际上是线性代数中分块矩阵乘法的直观实现。

以下是该方法的实现代码：

```cpp
#include <cuda.h>

#define TILE_SIZE (16)

template<typename _T> // 读取Pitch二维数组的函数，返回指向需要访问元素的指针。按行储存。
__device__ inline _T * Pitch2DMemPtr(_T * BaseAddress, size_t Row, size_t Column, size_t pitch)
{
	return (_T*)((char*)BaseAddress + Row * pitch) + Column;
}

// Matrix A*B
template<typename _T> //使用模板同时以便适配 int/半精度数/float/double 等计算需求
__global__ void MatMat(_T * MatA, const size_t MatAHeight, const size_t MatAWidth, const size_t MatAPitch,
	_T * MatB, /* MatBHeight = MatAWidth */ const size_t MatBWidth, const size_t MatBPitch,
	_T * MatR, size_t const MatRPitch) // ResultHeight = A.Height, ResultWidth = B.Width
	// 输入矩阵最好都用cudaMallocPitch分配，以避免内存不对齐的消耗
{
	__shared__ _T SubA[TILE_SIZE][TILE_SIZE];
	__shared__ _T SubB[TILE_SIZE][TILE_SIZE];

	// index in TILE
	const unsigned int x = threadIdx.x;
	const unsigned int y = threadIdx.y;

	// locate target block in result matrix
	const unsigned int GlobalX = blockIdx.x*TILE_SIZE + x;
	const unsigned int GlobalY = blockIdx.y*TILE_SIZE + y;
	// range equals to TILE_SIZE
	// 本 block 需要处理的范围是 [GlobalX : GlobalX+TILE_SIZE][GlobalY : GlobalX+TILE_SIZE] 这一块
	
	_T SubR = 0;	//用于储存该线程最终结果的变量
	// slide tiles over MatAWidth and MatBHeight
	for (size_t t = 0; t < MatAWidth; t += TILE_SIZE)
	{
		{
			// load SubA, element-by-element 按正常方式加载A中的元素
			SubA[y][x] = (t + x < MatAWidth && GlobalY < MatAHeight) ? (*Pitch2DMemPtr(MatA, GlobalY, t + x, MatAPitch)) : 0;

			// load SubB and transform in the same time 从B中加载数据并同时转置再储存到共享缓存中，这样再访问共享缓存时，访问可以合并
			SubB[x][y] = (GlobalX < MatBWidth && t + y < MatAWidth) ? (*Pitch2DMemPtr(MatB, t + y, GlobalX, MatBPitch)) : 0;
		}
		__syncthreads();	//等待所有线程加载完毕

		// multiply and add
		#pragma unroll	// 对固定大小的循环进行循环展开，进一步提高性能
		for (size_t i = 0; i < TILE_SIZE; ++i)
		{
			SubR += (SubA[y][i] * SubB[x][i]);
		}

		// 防止有的线程还在用共享内存计算，另一些已经跑到下一个循环加载新的数据了。
		// 这是一个比较严重的性能损失点，详见下文
		__syncthreads();
	}

	// write result
	if (GlobalX < MatBWidth && GlobalY < MatAHeight)
	{
		*Pitch2DMemPtr(MatR, GlobalY, GlobalX, MatRPitch) = SubR;
	}	
}
```

类似这段的代码在网上有关矩阵优化的文章中出现的很多。这种方法也确实能达到不错的速度（普通CPU的数百倍）。然而，估计一下GPU的理论计算能力可以发现，这一方法的成绩相对于峰值能力仍是九牛一毛。以**GTX TITAN X**为例，峰值运算能力为`6.6 TFLOPS`，而上述代码在**GTX TITAN X**上的表现不过`330 GFLOPS`，可以说只利用了`1/20`的计算能力。

![发现这一点时我的心情](/img/in-post/2016-09-01-CUDA-matrix-multiply-optimization/元首.jpg)

# 风辰书中的优化方案

在《科学计算与企业级应用并行优化》一书中（`P77`），提供了一段“可以达到**GTX 750Ti** 75% 理论峰值性能”的代码。可惜这段代码没有提供注释，对于初次接触CUDA性能优化菜鸟来说很难看懂。

这段代码的思路与上面一脉相承，也是线程们合力加载`A`的一横条、`B`的一纵条到共享储存器中，然后每个`block`计算结果矩阵`C`中的一个子块。从这一点看，在`block`负责计算的元素大小相等的前提下，风辰的算法与上面的TILE算法从`__global__`中加载数据的次数是一样的。既如此，性能改善从何而来？

在上一段代码中，我在第二个`__syncthreads()`处加了注释：

```cpp
	// 防止有的线程还在用共享内存计算，另一些已经跑到下一个循环加载新的数据了。
	// 这是一个比较严重的性能损失点，详见下文
	__syncthreads();
```

这一处性能损失的根源在于：TILE算法的流程是“加载-计算-加载-……”的循环。而每一次加载，都要等待上一次计算流程中所有的线程完全执行完毕，否则新加载/覆盖进来的数据，将进入那些尚在执行计算的线程，造成计算结果不正确。这意味着一轮计算过程中，从处理器到`__global__`的通讯资源必须是空闲的。理论上，访问一次`__global__`所需的时间是进行一次运算的数百倍，故CUDA程序设计中需要注意的一点便是：让访存与计算同步进行，以隐藏访问`__global__`的延迟。

![隐藏延迟](/img/in-post/2016-09-01-CUDA-matrix-multiply-optimization/hide-memory-latency.png)

上图概念性地解释了隐藏延迟的概念。多个线程交替进行计算与访存，等待访存结果时，切换另一个`context`执行，执行到等待访存时再换一个。直到所有线程都在等待访存指令返回时，这里的等待时间才被真的浪费掉了。

直观上看，这一做法与`__shared__`的用法有些矛盾，因为使用共享储存器的思路是每个线程都使用了别的线程加载的数据，在使用之前当然要等待它们运行完毕，于是根本没法用上面的思路隐藏延迟。看到书中的代码后，我发现实际上办法是还有的：使用寄存器作为共享内存的缓存。这种方式下数据流向为：

全局内存 --> 寄存器 --> 共享储存 --> 寄存器 --> 计算，循环（回到该行七点），累加 --> 写入结果

这样的做法看似复杂，但是实质上更快，因为这时候只需要在写入共享储存的前后执行`__syncthreads()`。而最慢的全局储存器与寄存器之间的访问，则由于有寄存器作为缓存，可以与计算同步进行，掩盖访存延迟。这一点如果不好理解，结合后面的代码理解。

用计算来掩盖访存延迟，显然需要足够的计算量，才能让处理器在等待期间不至于无事可做。如果仍按照一个线程处理一个元素的节奏，显然一次乘法和一次加法不足以喂饱处理器。正确的姿势是每个线程处理多个元素，以留出足够的时间去取下一轮运算需要的数据。

比较直观的想法是每个线程处理一个小方块，在小方块内部用一个二重循环逐个处理（此“处理”包括访存和运算）。这样带来了另一个问题：一个`block`负责的数据被多个线程各自为政分而治之，对全局储存器的访问不连续了。以下图为例，一维情况下，用4个线程（以赤橙黄绿表示）处理12个元素，则每个线程需要读取3个元素。按红色线程处理`0~2`，2号线程处理`3~5`这种直观方式来分配的话，三次读取动作的分配如图，显然每一次都是不连续的。

![不连续访问](/img/in-post/2016-09-01-CUDA-matrix-multiply-optimization/memory-access-discontinuous.png)

从图上看，解决方案也很简单，就是保证每一次读取时4个线程的目标紧挨着，也是三次处理完毕，如下图：

![连续访问](/img/in-post/2016-09-01-CUDA-matrix-multiply-optimization/memory-access-continuous.png)

第二幅图看起来简单，但是如果实现不知道，仅凭代码去猜测访问顺序，难度不小（起码我被困了一个下午）。二维的情况与此类似。画图比较麻烦，自行想象。

说了这么多，上代码：

（需要注意，这段代码认为矩阵`A`已经转置过了，也就是按列优先储存。将`A`转置可以更好地符号内存对齐要求。）

```cpp
#include "Pitch2DMemPtr.h"

// Thread configure: each thread in each loop take a (MARange x 1) range from MatA, and (1 x MBRange) range form MatB
// Scan and calculate over a range of MatAWidth (MatBHeight) to generate a (MARange x MBRange) range in result
// To keep memory access continuous, the above mentioned (MARange x 1) and (1 x MBRange) and (MARange x MBRange) range are discontinuous
// MatTransMulMat means MatA shall be transposed firstly. MatAHeight, MatAWidth are for original MatA but MatAPitch is for transposed MatA
template<unsigned int BlockSize, unsigned int MARange, unsigned int MBRange, typename _T>
__global__ void MatTransMulMat(_T * MatATrans, const size_t MatAHeight, const size_t MatAWidth, const size_t MatAPitch,
	_T * MatB, /* MatBHeight = MatAWidth */ const size_t MatBWidth, const size_t MatBPitch,
	_T * MatR, size_t const MatRPitch)
{
	extern __shared__ _T SCache[];	//使用线性而非二维的共享储存器。存疑，对性能有提升吗？
	
	_T * ACache = SCache;
	_T * BCache = ACache + MARange * BlockSize * BlockSize;
	// Malloc size of SCache shall be (MARange + MBRange) * BlockSize * BlockSize
	// 将线性内存的指针分解成两个使用

	const unsigned int tidX = threadIdx.x;
	const unsigned int tidY = threadIdx.y;

	// malloc result 用于累加
	_T R[MARange][MBRange] = { {0} };

	// Scan block-by-block
	// 类似于TILE方法的扫描，但是使用一套坐标转换机制保证了内存访问连续，具体看下面
	#pragma unroll 1
	for (size_t b = 0; b < MatAWidth; b += BlockSize)
	{
		// Each thread load MARange elements form MatA and MBRange elements form MatB to register
		// Coordinate are interlaced to keep aligned memory access
		_T TempA_G2S[MARange];
		#pragma unroll
		for (unsigned int i = 0; i < MARange; ++i)
		{
			// row and col value is swapped in below statement to fit transposed MatA
			// 访问的row坐标实际上应该是 (MARange*BlockSize)*blockIdx.y + i*BlockSize + tidX
			// 为了计算速度合并了一个同类项，造成程序更难理解了
			// MatATrans是已经转置过的
			// 访问坐标：指针指向第 b + tidY 行，然后向右移动 (MARange*BlockSize)*blockIdx.y 到本block负责的块
			// 再向右移动 i*BlockSize，到本次循环所需要处理的区域，再向右移动 tidX，到这一区域中本线程负责的元素
			// 参考上面关于“连续访问”的图理解
			TempA_G2S[i] = *Pitch2DMemPtr(MatATrans, b + tidY, (MARange*blockIdx.y + i)*BlockSize + tidX, MatAPitch);
		}
		_T TempB_G2S[MBRange];
		#pragma  unroll
		for (unsigned int i = 0; i < MBRange; ++i)
		{
			// row and col value is swapped in below statement to fit transposed MatA
			TempB_G2S[i] = *Pitch2DMemPtr(MatB, b + tidY, (MBRange*blockIdx.x + i)*BlockSize + tidX, MatBPitch);
		}

		__syncthreads();	// In case some thread write to shared memory while others are still using it in last loop

		// Write to shared cache
		// 读写的坐标始终是在交错模式下，保证对 __global__ 和 __shared__ 的访问一直是连续的
		#pragma unroll
		for (unsigned int i = 0; i < MARange; ++i)
		{
			// Clearer expression is (MARange*BlockSize)*tidY + i*BlockSize + tidX, but below is faster
			ACache[(MARange*tidY + i)*BlockSize + tidX] = TempA_G2S[i];
		}

		#pragma unroll
		for (unsigned int i = 0; i < MBRange; ++i)
		{
			BCache[(MBRange*tidY + i)*BlockSize + tidX] = TempB_G2S[i];
		}

		__syncthreads();

		// calculate and add to result
		#pragma unroll
		for (unsigned int i = 0; i < BlockSize; ++i)
		{
			// Load required data from shared memory to register.
			_T TempA[MARange];
			#pragma unroll
			for (unsigned int ia = 0; ia < MARange; ++ia)
			{
				TempA[ia] = ACache[(MARange*i + ia)*BlockSize + tidY];
			}
			
			_T TempB[MBRange];
			#pragma  unroll
			for (unsigned int ib = 0; ib < MBRange; ++ib)
			{
				TempB[ib] = BCache[(MBRange*i + ib)*BlockSize + tidX];
			}

			// Calculate and add
			#pragma unroll
			for (unsigned int ia = 0; ia < MARange; ++ia)
			{
				#pragma  unroll
				for (unsigned int ib = 0; ib < MBRange; ++ib)
				{
					R[ia][ib] += TempA[ia] * TempB[ib];
				}
			}
		}
	}

	// Write result
	// Calculate and add
	#pragma unroll
	for (unsigned int ia = 0; ia < MARange; ++ia)
	{
		#pragma  unroll
		for (unsigned int ib = 0; ib < MBRange; ++ib)
		{
			*Pitch2DMemPtr(MatR, (MARange*blockIdx.y + ia)*BlockSize + tidY, (MBRange*blockIdx.x + ib)*BlockSize + tidX, MatRPitch)
				= R[ia][ib];
		}
	}
}
```

下图展示了`block`中一个线程所负责写入到结果中的元素（红色）以及其需要读取的元素（绿色）。

![线程结构](/img/in-post/2016-09-01-CUDA-matrix-multiply-optimization/thread-configure.png)