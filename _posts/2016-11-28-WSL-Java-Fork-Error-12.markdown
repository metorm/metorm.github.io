---
layout: post
title:  WSL中Java代码无法调用任何本地命令的问题（error 12）
date:   2016-11-28 20:51:35 +0800
tags: [折腾,Linux,WSL]
---

Win 10 周年纪念版开始提供了一个基于 Ubuntu 的 Linux 子系统，可谓喜大普奔。本来只是一个命令行工具，结果更有高人发现简单设置即可在其中运行GUI程序乃至完整的桌面环境，惊为天人。日前（2016年11月28日），Insider preview 版本的 WSL 又升级到了 Ubuntu 16.04，并且提供了中文环境。

于是前天就在其中装了个 Xfce4，手感确实不错。得寸进尺心态之下，我在 Xfce4 中装了个 pyCharm，用来调试 tensorflow 这个只提供 Linux 版本的家伙。装好发现Java程序手感还是比较肉，这个可以理解。但随后我就发现了更大的问题：pyCharm识别不了python解析器，也无法从GUI中执行python文件。

简而言之，就是不给调用任何本地程序，油盐不进。有详细错误提示的地方，就提示“执行失败。无法分配内存。error = 12, cannot allocate memory”

难道是真缺内存？一番搜索，把分配给pyCharm自带JVM的内存从350M扩大到1G，直观上感觉操作不那么肉了，但是这个症状依旧。看来不是直接的内存问题。那么，是权限或者JVM问题？然而权限查不出什么问题，我尝试了用openjdk或者oracle jdk替换自带的jre，结果一样。

折腾半个下午之后，我开始怀疑pyCharm太娇气了，决定换一个试试，于是罩了同样是基于Java的liclipse，运行，结果一样。一不做二不休，去装了个Sublime+Anaconda插件，试运行，居然正常！

于是情况就比较明朗了：JVM在这里无法进行系统调用！马上测试：

```
import java.io.IOException;
public class prova {
    public static void main(String[] args) throws IOException {
                Runtime.getRuntime().exec("ls");

    }
}
```

不出所料：

```
Exception in thread "main" java.io.IOException: Cannot run program "ls": error=12, 无法分配内存
	at java.lang.ProcessBuilder.start(ProcessBuilder.java:1048)
	at java.lang.Runtime.exec(Runtime.java:620)
	at java.lang.Runtime.exec(Runtime.java:450)
	at java.lang.Runtime.exec(Runtime.java:347)
	at prova.main(prova.java:6)
Caused by: java.io.IOException: error=12, 无法分配内存
	at java.lang.UNIXProcess.forkAndExec(Native Method)
	at java.lang.UNIXProcess.<init>(UNIXProcess.java:247)
	at java.lang.ProcessImpl.start(ProcessImpl.java:134)
	at java.lang.ProcessBuilder.start(ProcessBuilder.java:1029)
	... 4 more
```

这就比较尴尬了。测试了openjdk和oracle jdk，结果相同。应该是WSL的问题，没跑。可是这种问题巨硬才能解决，不像是我等升斗小民可以hold的。最后一招，去github上WSL的issue系统发帖，询问Java的fork为啥不能执行。居然很快就有官方人员回应，其中的对话就不贴了，关键问题在于：

>I assume (though I don't know for sure) that the memory limitation is due to the difference in the behavior of the NT kernel vs the Linux kernel:

>Windows assumes that, if an application maps ("asks the kernel to let it use in the future") a block of virtual-memory pages, it does actually intend to use them eventually. It therefore requires that enough memory currently be available, either as real physical memory or as swap, to allow the entirety of any given mapping to be allocated and to have data assigned to it. If it gets some unmanageably-large allocations (I'm sure there are specific rules but I don't know the rules offhand), it doesn't necessarily grow the page file; it may instead choose to return an error code to indicate to the application that it probably doesn't want to be allocating that much memory on this machine. This approach can be a little clunky/limiting, but it has some nice correctness guarantees and it tends to encourage good application behavior.

>Linux pretty much always lets applications map whatever they want, whether or not it's reasonable on the current machine. Therefore, for simplicity, some applications' runtimes (notably including Java; I'm not personally aware of any other major applications that do this to such a large extent) tend to map giant chunks of memory up front, presumably so that they don't have to think about mapping more memory later. They then allocate and use only as much memory as they actually use. If they allocate more memory than is available on the system (and the swap file runs out of space, etc), which they can do because the kernel has already promised the mapping, at that point it's too late for the kernel to provide an error code; the only correct thing it can do is to force-kill the application.

>Supporting the Linux behavior on Windows would therefore require either allowing the Windows pagefile to grow much larger than the amount of memory that Java is likely to actually use, just in case it does use it (which would be problematic on machines like yours with limited disk space), or implementing the Linux out-of-memory killer in Windows, which I would expect to require a change to the core NT kernel (not WSL-specific code).

>(Disclaimer: I'm not actually a WSL dev, just a Windows and Linux user; this is just my best guess at the cause based on the behavior of the two kernels' public APIs, it's certainly possible that there's a different limitation going on under the hood.)

> [Java error=12, cannot allocate memory (also chmod)](https://github.com/Microsoft/BashOnWindows/issues/1403)

简而言之，就是内存映射问题，两个系统的处理方式不一致。这个估计还要巨硬的人死几斤脑细胞来彻底解决。不过临时解决方案导师很简单：更改Windows下的页面文件设置，将虚拟内存容量固定为一个不小于物理内存的值（可怜我的SSD啊），问题即可解决。实测通过！