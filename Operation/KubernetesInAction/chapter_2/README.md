# Chapter 2 First steps with Docker and Kubernetes

## 2.1 Creating, running, and sharing a container image

Omit.

## 2.2 Setting up a Kubernetes cluster

The simplest and quickest path to fully functioning Kubernetes cluster
is by using Minikube. Minikube is a tool that sets up a single-node cluster
that's great for both testing Kubernetes and developing apps locally.

## 2.3 Running your first app on Kubernetes

### 2.3.1 Deploy Node.js app

The simplest way to deploy app is to use the `kubectl run` command, which
will create all the necessary components without having to deal with
JSON or YAML.

```sh
kubectl run kubia --image=luksa/kubia --port=8080
```

A pod called `kubia` has been created.

#### Introducing Pods

A pod is a group of one or more tightly related containers that will
always run together on the same worker node and in the same Linux namespace.

To better understand the relationship between containers, pods, and
nodes, examine below. As you can see, each pod has its own IP and
contains one or more containers, each running an application process.

![The relationship between containers, pods, and physical worker nodes](https://s2.loli.net/2022/08/19/rHPp85JWLBVYAax.png)

#### Listing pods

You can use `kubectl get pods` to list the pods. To see more information
about the pod, you can also use the `kubectl describe pod` command.

#### Understanding what happened behind the scenes

When you ran the `kubectl` command, it created a new ReplicationController object
in the cluster by sending a REST HTTP request to the Kubernetes API server.
The ReplicationController then created a new pod, which was then
scheduled to one of the worker nodes by the Scheduler. The Kubelet
on that node saw that the pod was scheduled to it and instructed Docker
to pull the specified image from the registry. After downloading the image,
Docker created and ran the container.

### 2.3.2 Accessing your web application

With your pod running, how do you access it? We mentioned that each
pod gets its own IP address, but this address is internal to
the cluster and isn't accessible from outside of it. To make the pod
accessible from the outside, you'll expose it through a Service object.
You'll create a special service of type `LoadBalancer`. By creating a
`LoadBalancer`-type service, an external load balancer will be created
and you can connect to the pod through the load balancer's public IP.

#### Creating a service object

```sh
kubectl expose pod kubia --type=LoadBalancer --name kubia-http
```

#### Listing services

```sh
kubectl get services
```

However, when using `minikube`, the status of EXTERNAL IP would be pending.
You can get the IP and port through which you can access the service
by running `minikube service kubia-http`.
