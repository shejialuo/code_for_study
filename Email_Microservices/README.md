# Email Microservice System

+ 编程语言：C++
+ 系统环境：Linux，你需要使用Linux环境下的g++。

为了保证脚本的正常执行，你需要解压在`/home/<username>/Projects/`下。

在`scripts`目录下有以下几个脚本。

+ `build_cpp.sh`：用于编译C++源文件。
  + `build_cpp.sh ALL`：用于编译全部的C++源文件。
  + `build_cpp.sh <微服务名>`：用于编译指定的C++源文件。
+ `create_network.sh`：用于生成Docker局域网。
+ `build_image.sh`：用于生成Docker镜像，用法同`build_cpp.sh`。
+ `build_container.sh`:用于生成Docker容器，用法同
`build.cpp.sh`。

初始运行软件：

+ `scripts/build_cpp.sh ALL`
+ `scripts/create_network.sh`
+ `scripts/build_image.sh ALL`
+ `scripts/build_container.sh ALL`
+ `cd Monitor` 然后执行 `./Monitor`

在运行的过程中，可以通过`sudo docker container logs <微服务名>`来看日志输出。
