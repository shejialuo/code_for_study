# Chapter 1 Introducing Kubernetes

## 1.1 Understanding the need for a system like Kubernetes

Omit.

## 1.2 Introducing container technologies

Omit.

## 1.3 Introducing Kubernetes

### 1.3.1 Understanding the architecture of a Kubernetes cluster

At the hardware level, a Kubernetes cluster is composed of
many nodes, which can be split into two types:

+ The *master* node, which hosts the *Kubernetes Control Plane* that
controls and manages the while Kubernetes system.
+ Worker *nodes* that run the actual applications you deploy.

Below shows the components running on these two sets of nodes.

![The components that make up a Kubernetes cluster](https://s2.loli.net/2022/08/19/H4xqVwLOv1ieMT3.png)

#### The control plane

The *Control Plane* is what controls the cluster and makes it function.
It consists of multiple components that can run on a single master node
or be split across multiple nodes and replicated to ensure high availability.
These components are

+ The *Kubernetes API Server*, which you and the other Control Plane
components communicate with.
+ The *Scheduler*, which schedules your apps.
+ The *Controller Manager*, which performs cluster-level functions,
such as replicating components, keeping track of worker nodes,
handling node failures, and so on.
+ `etcd`, a reliable distributed data store that persistently stores
the cluster configuration.

#### The nodes

The worker nodes are the machines that run your containerized applications.
The task of running, monitoring, and providing services to your applications
is done by the following components:

+ Docker, rkt, or another *container runtime*, which runs your containers.
+ The *kubelet*, which talks to the API server and manages containers on its node.
+ The *Kubernetes Service Proxy (kube-proxy)*, which load-balances
network traffic between application components.

### 1.3.2 Running an application in Kubernetes

To run an application in Kubernetes, you first need to package it up
into one or more container images, push those images to an image registry,
and then post a description of your app to the Kubernetes API server.

The description includes information such as the container image
or images that contain your application components,
how those components
are related to each other, and which ones need to be run co-located
(together on the same node) and which don't. For each component, you
can also specify how many copies you want to run.

#### Understanding how the description results in a running container

When the API server processes your app's description, the Scheduler
schedules the specified groups of containers onto the available worker
nodes based on computational resources required by each group and
the unallocated resources on each node at that moment. The kubelet
on those nodes then instructs the Container Runtime to pull the
required container images and run the containers.

![A basic overview of the Kubernetes architecture and an application running on top of it](https://s2.loli.net/2022/08/19/CI1SVFEatZmzkNw.png)

#### Keeping the containers running

Once the application is running, Kubernetes continuously make sure that
the deployed state of the application always matches the description
you provided.

#### Scaling the number of copies

While the application is running, you can decide you want to increase or
decrease the number of copies, and Kubernetes will spin up additional
ones or stop the excess ones, respectively.
