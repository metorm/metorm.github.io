---
layout: post
title:  "bwdistsc 快速距离场计算函数解析!"
date:   2015-6-27 22:04:00 +0800
tags: [数学,距离场]
---

# bwdistsc 快速距离场计算函数解析

之前的《距离场计算：维度诱导法（dimension-induction）的基本原理》，已经简述了快速距离场计算基本思路，[bwdistsc函数](http://uk.mathworks.com/matlabcentral/fileexchange/15455-3d-euclidean-distance-transform-for-variable-data-aspect-ratio)是 Yuriy Mishchenko 对上述思路的实现。这里将函数源码做分析与解释。介意结合 [Yuriy Mishchenko 论文](http://link.springer.com/article/10.1007%2Fs11760-012-0419-9)阅读。

经测试，函数速度相当快，并且计算量与实际问题复杂度相关。若实际问题中物体较为规则，在相同网格数量下，算法执行速度将比复杂拓扑结构物体相比有明显提高。

*文章为简明删去了部分原有注释，主要是权利声明等。需要使用函数的话可到上文链接处Mathworks网站下载最新版*

## 声明与输入解析

```matlab
function D=bwdistsc(bw,aspect)

% parse inputs
if(nargin<2 || isempty(aspect)) aspect=[1 1 1]; end

% determine geometry of the data
if(iscell(bw)) shape=[size(bw{1}),length(bw)]; else shape=size(bw); end

% correct this for 2D data
if(length(shape)==2) shape=[shape,1]; end
if(length(aspect)==2) aspect=[aspect,1]; end
    
% allocate internal memory
D=cell(1,shape(3)); for k=1:shape(3) D{k}=zeros(shape(1:2)); end
```

>以上都是预处理。二维还是三维，网格长宽比是否为1，等等。

## 计算距离场

```matlab

%%%%%%%%%%%%% scan along XY %%%%%%%%%%%%%%%%
for k=1:shape(3)    %循环，按第三个坐标（z）循环
    if(iscell(bw)) bwXY=bw{k}; else bwXY=bw(:,:,k); end
    %切片储存在bw中
        
    % initialize arrays
    DXY=zeros(shape(1:2));
    D1=zeros(shape(1:2));

    % if can, use 2D bwdist from image processing toolbox    
    if(exist('bwdist') && aspect(1)==aspect(2))
        D1=aspect(1)^2*bwdist(bwXY).^2;
    else    % if not, use full XY-scan
```

### 计算一维距离场

切成一条一条的计算。一行一行，从第一行向最后一行扫略（从上到下），再从最后一行向第一行扫略（从下到上），从而生成了两个记录矩阵。这两个矩阵中每一个网格点记录的值是当前列中，按当前方向扫略的情况下，离自己最近的物体网格点。

>下文中"on"-pixel指的是输入bw中数值为1的网格，即认为是有物体的网格
>虽然有很多“条”，但是都用了向量化计算，所以这里没有循环语句

```matlab

        % z的循环还在进行
        %%%%%%%%%%%%%%% X-SCAN %%%%%%%%%%%%%%%        
        % reference for nearest "on"-pixel in bw in x direction down
        
        %  scan bottow-up (for all y), copy x-reference from previous row 
        %  unless there is "on"-pixel in that point in current row, then 
        %  that is the nearest pixel now
        xlower=repmat(Inf,shape(1:2)); 
        
        xlower(1,find(bwXY(1,:)))=1;    % fill in first row
        for i=2:shape(1)
            xlower(i,:)=xlower(i-1,:);  % copy previous row
            xlower(i,find(bwXY(i,:)))=i;% unless there is pixel
        end
        
        % reference for nearest "on"-pixel in bw in x direction up
        xupper=repmat(Inf,shape(1:2));
        
        xupper(end,find(bwXY(end,:)))=shape(1);
        for i=shape(1)-1:-1:1
            xupper(i,:)=xupper(i+1,:);
            xupper(i,find(bwXY(i,:)))=i;
        end
                
        % build (X,Y) for points for which distance needs to be calculated
        idx=find(~bwXY); [x,y]=ind2sub(shape(1:2),idx);
        
        % update distances as shortest to "on" pixels up/down in the above
        % 将两个矩阵合并起来
        DXY(idx)=aspect(1)^2*min((x-xlower(idx)).^2,(x-xupper(idx)).^2);
```

### 计算二维距离场

``` matlab

        % z的循环还在继续
        %%%%%%%%%%%%%%% Y-SCAN %%%%%%%%%%%%%%%
        % this will be the envelop of parabolas at different y
        D1=repmat(Inf,shape(1:2));
        
        p=shape(2);
        for i=1:shape(2)
            % some auxiliary datasets
            % 取出平面上的一列。从左向右按列扫描
            % d0中存放的是距离的平方
            d0=DXY(:,i);
            
            % selecting starting point for x:
            % * if parabolas are incremented in increasing order of y, 
            %   then all below-envelop intervals are necessarily right-
            %   open, which means starting point can always be chosen 
            %   at the right end of y-axis
            % * if starting point exists it should be below existing
            %   current envelop at the right end of y-axis
            dtmp=d0+aspect(2)^2*(p-i)^2;
                %均匀网格下，aspect的三个分类分量均为1，可以忽略。写在这里真影响理解……

            L=D1(:,p)>dtmp;    % 比较D1中的一列与dtmp的大小
                %一维数组L中储存的是大小比较的结果量（0或1，认为是bool）
            idx=find(L);            
            D1(idx,p)=dtmp(L);    % D1的最后一列被设置为dtmp的一部分（取最小值）
            % 上面这几句还可以精简
          

            % these will keep track along which X should 
            % keep updating distances            
            map_lower=L;
            idx_lower=idx;    %储存了dtmp比D1小的那些位置
            
            % scan from starting points down in increments of 1
            % 从尾巴开始向前扫描，重复类似上面的比较
            % 但是只关心lower位置，下面有解释
            for ii=p-1:-1:1
                % new values for D
                dtmp=d0(idx_lower)+aspect(2)^2*(ii-i)^2;
                
                % these pixels are to be updated
                L=D1(idx_lower,ii)>dtmp;
                D1(idx_lower(L),ii)=dtmp(L);
                
                % other pixels are removed from scan
                map_lower(idx_lower)=L;                
                idx_lower=idx_lower(L);
                
                if(isempty(idx_lower)) break; end
            end
        end
    end
    D{k}=D1; 
end     %z的循环结束了
```

>从尾巴开始向前扫描，重复类似上面的比较，但是**只关心已经包含在lower中的位置**。循环中不断缩小lower区域，lower若为空则可以提前结束循环

#### 原因解释

参考论文中引理1,抛物线比下包络线低的位置只可能存在于一个区间，而不会断断续续的存在于多个区间。又由于所有的抛物线二次项系数相等，所以这一区间必定为无穷区间$$(-\infty,x)$$或者$$(x,\infty)$$或者$$(-\infty,\infty)$$（排除完全重合的情况）。如果是$$(x,\infty)$$的情况，则下面这个从右向左扫描，找到包络线与抛物线交点即不再继续的算法容易理解。

由于上面的i循环是从左往右扫描的，新加入的抛物线要么在包络线右半部分上升段与之相交（上述区间为$$(x,\infty)$$），或者整条都低于包络线（上述区间为$$(-\infty,\infty)$$）。**$$(-\infty,x)$$的情况不存在**，因为它已经被扫描方法排除了。可以画图观察。

### 计算三维距离场
```matlab
%%%%%%%%%%%%% scan along Z %%%%%%%%%%%%%%%%
% 新建一个跟D一样大的元胞数组D1，用Inf填充
D1=cell(size(D));
for k=1:shape(3) 
  D1{k}=repmat(Inf,shape(1:2)); 
end

% start building the envelope 
p=shape(3);
for k=1:shape(3)
    % if there are no objects in this slice, nothing to do
    if(isinf(D{k}(1,1)))    %有一个是Inf，即说明这一层里面一个物体点都没找到
      continue;
    end
```

下面的算法，跟由一维构建二维方式相同。由于每次在一个维度上做文章，所以无论扩展到多少维度，需要计算的总是只有那么一小撮抛物线，所以算法基本一样。需要的话都能写出递归来。

```matlab

    % selecting starting point for (x,y):
    % * if parabolas are incremented in increasing order of k, then all 
    %   intersections are necessarily at the right end of the envelop, 
    %   and so the starting point can be always chosen as the right end
    %   of the axis
    
    % check which points are valid starting points, & update the envelop
    dtmp=D{k}+aspect(3)^2*(p-k)^2;
    L=D1{p}>dtmp; 
    D1{p}(L)=dtmp(L);    
    
    % map_lower keeps track of which pixels can be yet updated with the 
    % new distance, i.e. all such XY that had been under the envelop for
    % all Deltak up to now, for Deltak<0
    map_lower=L;
        
    % these are maintained to keep fast track of whether map is empty
    idx_lower=find(map_lower);
    
    % scan away from the starting points in increments of -1
    for kk=p-1:-1:1
        % new values for D
        dtmp=D{k}(idx_lower)+aspect(3)^2*(kk-k)^2;
                    
        % these pixels are to be updated
        L=D1{kk}(idx_lower)>dtmp;
        map_lower(idx_lower)=L;
        D1{kk}(idx_lower(L))=dtmp(L);
                    
        % other pixels are removed from scan
        idx_lower=idx_lower(L);
        
        if(isempty(idx_lower)) break; end
    end
end
% prepare the answer
if(iscell(bw))
    D=cell(size(bw));
    for k=1:shape(3) D{k}=sqrt(D1{k}); end
else
    D=zeros(shape);
    for k=1:shape(3) D(:,:,k)=sqrt(D1{k}); end
end
end
```

>计算完成