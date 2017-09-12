# distribute-of-tensorflow-
this is a litttle demo of distribute tensorflow 
本代码修改自https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dist_test/python/mnist_replica.py
基于Tensorflow的CNN分布式例子。运行的数据集是mnist-data，

同时启动三台服务器，ip分别为192.168.80.130，192.168.80.131，192.168.80.132

运行方式：

分别在三台机器上运行下面三行：

python3 cnn_dist.py --job_name="ps" --task_index=0

python3 cnn_dist.py --job_name="worker" --task_index=0

python3 cnn_dist.py --job_name="worker" --task_index=1

注：

1.由于实验设备没有显卡，本代码采用cpu运行，如果需要修改GPU运行，则将0改为1.
flags.DEFINE_integer("num_gpus", 0,

                     "Total number of gpus for each machine."
                     
                     "If you don't use GPU, please set it to '0'")

2.可以根据数据集的实际需求更改同步或者异步并行化，默认异步，同步只需将False改为True即可。
 flags.DEFINE_boolean("sync_replicas", False,
 
                     "Use the sync_replicas (synchronized replicas) mode, "
                     
                     "wherein the parameter updates from workers are aggregated "
                     
                     "before applied to avoid stale gradients")
                     
3.请根据实际情况更改Ip地址，端口号为2222，可以根据实际情况选取合适端口

ps服务器：

flags.DEFINE_string("ps_hosts","192.168.80.130:2222",

                    "Comma-separated list of hostname:port pairs")
                    
flags.DEFINE_string("worker_hosts", "192.168.80.131:2222,192.168.80.132:2222",

                    "Comma-separated list of hostname:port pairs")                    
                 



