# 融合系统的UML类图

smh

2019.01.09

## UML类图中的关系

* 继承

  继承是类与类之间或者接口与接口之间的关系。又被称为泛化。

* 实现

  实现是类与接口类之间的关系。

* 关联

  对于两个相对独立的对象，当一个对象的实例与另一个对象的一些特定实例存在固定的对应关系时，这两个对象之间为关联关系。

  表示类与类之间的联接，由双向关联和单向关联。

  关联关系以实例变量的形式存在，在每一个关联的端点，还可以有一个基数，表明这一端点的类可以有几个实例。

  体现在代码上：双向关联中两个类都有对方的一个指针，当然也可以是引用或者值。关联关系是使用实例变量来实现。

  与后面*聚合*与*组合*的区别在于这里两个类之间是平等的关系。

* 聚合

  关联关系的一种，是强的关联关系。**聚合是整体和个体的关系**。关联关系与聚合关系在语法上是没办法区分的，从语义上才能更好的区分两者的区别。

  聚合关系也是通过实例变量实现的。当类之间由整体-部分关系的时候，我们就可以使用聚合或者聚合。

  注意这里涉及到了整体与个体的关系。

* 组合(合成)

  组合关系也是关联关系的一种，是比聚合关系更强的关系。聚合关系是不能共享的。

  表示类之间整体和部分的关系，组合关系中部分和整体具有统一的生存期。一旦整体对象不存在，部分对象也将不存在。

  部分对象与整体对象之间具有共生死的关系。

* 依赖

  对于两个相对独立的对象，当一个对象负责构造另一个对象的实例，或者依赖另一个对象的服务时，这两个对象之间主要体现为依赖关系。

  与关联不同的是，**依赖关系是以参数变量的形式传入到依赖类中的**，依赖是单向的，要避免双向依赖。一般来说，不应该存在双向依赖。

  依赖是一种弱关联，只要一个类用到另一个类，但是和另一个类的关系不太明显时(如不是成员)，就可以把这种关系看成依赖。

  体现在代码上： **依赖关系表现在局部标量、方法的参数以及对静态方法的调用等**。

* 聚合与组合

  * 聚合与组合是一种关联关系，只是额外具有整体-部分的意涵。

  * 部件的生命周期不同

    在聚合关系中，整体不会拥有个体的生命周期，所以整件删除时，个体不会被删除。再者，多个整体可以共享同一个个体。体现在代码上就是个体以参数的方式传递到整体的构造函数中。

    在组合关系中，整体会拥有个体的生命周期，所以整体删除时，个体也会被删除。再者，多个整体不能同事共享同一个整体。体现在代码上就是个体的实例化在狗在函数内完成定义，而不是整体类的外部。

  * 聚合关系是'has-a'关系，组合关系是'contains-a'关系

* 关联和聚合

  * 表现在代码层面，和关联关系是一致的，只能从语义上区分

  * 关联和聚合的区别主要在语义上，关联的两个对象之间一般是平等的，聚合一般不是平等的

  * 关联是一种结构化的关系，之一种对象和另一种对象有联系

  * 关联和聚合是视问题域而定的

* 关联和依赖

  * 关联关系中，体现的是两个类、或者类与接口之间的语义级别的一种强依赖关系，不存在依赖关系的偶然性、关系也不是临时性的，一般是长期性的，而且双方的关系一般是平等的

  * 依赖关系中，可以简单的理解，就是一个类A使用了另一个类B，而这种使用关系具有偶然性，临时性的、非常弱的，但是B类的变化影响到A类。

* 综合比较

  这几种关系都是语义级别的，所以从代码层面并不能完全区分各种关系，但总的来说，几种关系表现的强弱程度依次为：

  组合 > 聚合 > 关联 > 依赖。


* 参考链接

  [1] [UML类图关系](https://www.cnblogs.com/alex-blog/articles/2704214.html)

  [2] [Graphic Design Patterns](https://design-patterns.readthedocs.io/zh_CN/latest/read_uml.html)

  [3] [浅谈UML中的聚合与组合](https://blog.csdn.net/liushuijinger/article/details/6994265)

## RDLW-Sal based fusion system 成员分析

* FusionSystem

  * 函数成员

    void doFusion(cv::Mat& imgOut, cv::Mat& imgInA, cv::Mat& imgInB);

    void setGFParams();

  * 变量成员

    RDLWavelet *mpRDLWavelet_;

    mpWeightedMap *mpWeightedMap_;

* RDLWavelet

  * 函数成员

    void doRDLWavelet();

    void doInverseRDLWavelet();

    私有函数成员

    void HorizontalPredict();

    void VerticalPredict();

    void HorizontalUpdate();

    void VerticalUpdate();

    void InverseHorizontalUpdate();

    void InverseVerticalUpdate();

  * 变量成员

* WeightedMap

  * 函数成员

    void setParams();

    void doWeightedMap();

    私有函数成员

    void localsaliency();

    void globalsaliency();

    void doSaliencyDetection();

  * 变量成员

    GuidedFilter *mpGuidedFilter_;

* GuidedFilter

  * 函数成员

    void doGuidedFilter();

    私有函数成员

    void bindTexture();

    void releaseTexture();

    void boxfilterImgI(); 

    void boxfilterImgP();

    void boxfilterCorrI();

    void boxfilterCorrIp();

## RDLW-Sal based Fusion System UML类图

* 各个类之间的关系应当是**组合**，理由如下：

  * 存在整体-与个体的关系，因此不是关联

  * 整体与个体之间生存周期一致，因此不是聚合

  综上所属，各个类之间应该是组合关系。

* 类图

  类图是显示出类、接口以及它们之间的静态结构与关系的图。其中最基本的单元是类或接口。

  类图一般分为几个部分：

  * 类名

    如果是正体字，则说明该类是一个具体的类；如果是斜体字，则说明该类是一个抽象类。

  * 属性列表

    属性可以是public, private, protect，可以分别用+、-、#代表。

  * 方法

    方法也可以用+、-、#分别表示public、private、protect。对于静态属性，属性名会加上一条下划线。

  此外，类图既能表示类之间的关系，还能表示对象之间的关系。二者的区别在于：对象图中对象名下面会加上一条下划线。

* 时序图

  是为了展示对象之间的交互细节。

  时序图是显示对象之间交互的图，这些对象是按照顺序排列的。时序图中现实的是参与交互的对象以及对象之间消息交互的顺序。

  时序图包括的建模元素由：对象(Actor)、生命线(Lifeline)、控制焦点(Focus of control)、消息(Message)等。
