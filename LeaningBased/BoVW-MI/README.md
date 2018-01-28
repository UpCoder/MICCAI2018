- 文章大意：将基础的BoVW进行了改造，引入了mutual information，也就是在原先的词典集合中选择出一些更相关的视觉单词，作为新的字典。
- 文章流程：
	- 首先文章介绍了基本的BoVW过程，详情可以参考我的这篇博客：[BoVW](http://blog.csdn.net/liangdong2014/article/details/70651239)
	- 其次对于特征的计算：文章中并没有对提取出的patch计算特殊的特征，而是利用PCA降维的方式来提取了特征。
	- 总结一下字典的生成过程，如下所示：
		- 提取ROI
		- 提取Patch
		- 计算特征向量
		- 使用PCA进行降维
		- 进行聚类，这里我们使用kmeans聚类方法
		- 得到K个质心点，即就是我们的字典集合。
	- 得到了整个字典的集合，我们怎么去进行分类呢？具体流程如下：
		- 输入图像
		- 提取patch
		- PCA降维
		- 从字典中查找，获得每个patch对应的单词
		- 获得特征向量
		- 使用SVM或是其他分类器进行分类
	- 上面就是文章中说道的普通的BoVW的方法，那么怎么使用mutual information呢？
		- 在我看来mutual information 就是在原有的词库中选择与分类结果更相关的单词，而提取与分类结果不相关的单词。用原文的话就是：Feature relevance is often characterized in terms of mutual information between the feature value and the class label. 比如说，我们要分类猫和狗，那么天空对于我们来说就是不重要的视觉单词，所以我们要将其剔除。
		- 那么我们怎么选择相关的单词呢？换句话说，我们怎么计算每个单词的相关程度呢？文中给出了如下的计算方式：
			- 每次单词的计算公式如下所示：
				![image](http://ocnsbpp0d.bkt.clouddn.com/formula1.jpg)
			- Yi 代表的就是第i个视觉单词，C代表的就是分类结果的集合，P的计算公式如下所示：
				![image](http://ocnsbpp0d.bkt.clouddn.com/formula2.jpg)
			- pi(v,c) 代表的是所有样本中，第i个视觉单词对应的等级为v且对应的分类结果为c的个数。
			- 举例如下：我们目前有两种类别，5个样本，3个视觉单词，数据如下所示：
				- 第一个样本：1 2 3， 0
				- 第二个样本：2 1 1， 1
				- 第三个样本：2 2 2， 0
				- 第四个样本：3 1 1， 1
				- 第五个样本：1 3 2， 1
				- 那么p0(1, 0) = 0.2;p0(2, 0) = 0.2;p0(3, 0) = 0;p0(1, 1) = 0.2;p0(2, 1) = 0.2;p0(3, 1) = 0.2;
				- 这样我们就可以计算出每个单词的相关程度
	- 计算出每个单词的相关程度后，我们就可以选择出最相关的m个视觉单词，构成新的特征向量，然后再去使用SVM分类器去分类。
- 实现代码：[code](https://github.com/UpCoder/BoVW-MI)
		
		