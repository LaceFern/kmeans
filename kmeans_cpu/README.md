
compare_kmeans.cpp实现了standard kmeans，minibatch kmeans（mbkm）和staleness-reduction kmeans(srmbkm，新算法)三种算法。

通过定义宏（DSINFO_NUM，K_NUM，BS_NUM，SEED_NUM，A_NUM），并给相应数组赋值（dsinfo_arr，k_arr，seed_arr，batchsize_arr，alpha_arr），来实现对三种算法的不同配置。其中，k_arr和seed_arr同时用于三个算法，batchsize_arr用于mbkm和srmbkm，alpha_arr只用于srmbkm。

另外代码也支持通过修改threads和threads_loss来修改分配和计算loss时的样本并行度，分配步骤和loss计算的维度并行由于使用SIMD指令而固定为16。
    
编译指令见下：

g++ -o compare_kmeans.run -pthread -mavx512f -mavx512f -mavx512cd -mavx512er -mavx512pf -mavx512vl -mavx512dq -mavx512bw compare_kmeans.cpp ./include/util/timer.cpp ./include/util/dataIo.cpp ./include/util/arguments.cpp ./include/util/allocation.cpp ./include/mckm/mckm.cpp ./include/cmdparser/cmdlineparser.cpp ./include/cmdparser/cmdlineparser.h ./include/logger/logger.h ./include/logger/logger.cpp -O3
