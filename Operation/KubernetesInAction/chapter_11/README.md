# Chapter 11 Understanding Kubernetes internals

## 11.1 Understanding the architecture

A Kubernetes cluster is split into two parts:

+ The Kubernetes Control plane.
+ The worker nodes.

The Control Plane is what controls and makes the whole cluster function:

+ The etcd distributed persistent storage.
+ The API server.
+ The Scheduler.
+ The Controller Manager.

+ The Kubelet.
+ The Kubernetes Service Proxy.
+ The container run time.

Beside the Control Plane components and the components running on the nodes,
a few add-on components are required for the cluster to provide everything
discussed so for. This includes

+ The Kubernetes DNS server.
+ The Dashboard
+ An Ingress controller.
+ Heapster.
+ The Container Network Interface network plugin.

### 11.1.1 The distributed nature of Kubernetes components

#### How these components communicate

Kubernetes system components communicate only with the API server. They don't
talk to each other directly. The API server is the only component that communicates
with etcd. None of the other components communicate with etcd directly, but
instead modify the cluster state by talking to the API server.

#### Running multiple instances of individual components

Although the components on the worker nodes all need to run on the same node,
the components of the Control Plane can easily be split across multiple servers.
There can be more than one instance of each Control Plane component running
to ensure high availability. While multiple instances of etcd and API server can
be active at the same time and do perform their jobs in parallel, only a single
instance of the Scheduler and the Controller Manager may be active at a given
time—with the others in standby mode.

#### How components are run

The Control Plane components, as well as the kube-proxy, can either be deployed
on the system directly or they can run as pods.

The Kubelet is the only components that always runs as a regular system component,
and it's the Kubelet that then runs all the other components as pods. To run the
Control Plane components as pods, the Kubelet is also deployed on the master.

### 11.1.2 How Kubernetes uses etcd

All the objects you've created need to be stored somewhere in a persistent manner
so their manifests survive API server restarts and failures. For this, Kubernetes
uses etcd.

The only component that talks to etcd directly is the Kubernetes API server. All
other components read and write data to etcd indirectly through the API server.

### 11.1.3 What the API server does

The Kubernetes API server is the central component used by all other components
and by clients. It provides a CURD interface for querying and modifying the cluster
state over a RESTful API. Below shows what happens inside the API server
when it receives the request.

![The operation of the API server](https://s2.loli.net/2023/08/09/WCMONKGiBsAwjvf.png)

First, the API server needs to authenticate the client sending the request.
This is performed by one or more authentication plugins configured in the API server.
The API server call these plugins in turn, until one of them determines who
is sending the request. It does this by inspecting the HTTP request.

Besides authentication plugins, the API server is also configured to use one or
more authorization plugins. Their job is to determine whether the authenticated
user can perform the requested action on the requested resource.

If the request is trying to create, modify, or delete a resource, the request is
sent to the Admission Control. Again, the server is configured with multiple
Admission Control plugins. These plugins can modify the resource for different
reasons. The resource passes through all plugins.

### 11.1.4 Understanding how the API server notifies clients of resource changes

The API server doesn't do anything except what we've discussed. The API server
doesn't even tell controllers what to do. All it does is enable those controllers
and other components to observe changes to deployed resources.

Client watches for changes by opening an HTTP connection to the API server.
Through this connection, the client will then receive a stream of notifications
to the watched objects. Every time an object is updated, the server sends
the new version of the object to all connected clients watching the object.

### 11.1.5 Understanding the Scheduler

The Scheduler would wait for newly created pods through the API server's watch
mechanism and assign a node to each new pod.

The Scheduler doesn't instruct selected node to run the pod. All the Scheduler does
is update the pod definition through the API server the notifies the Kubelet.

### 11.1.6 Introducing the controllers running in the Controller Manager

The single Controller Manager process currently combines a multitude of controllers
performing various reconciliation tasks. Eventually, those controllers will
be split up into separate processes, enabling you to replace each one with a
custom implementation if necessary. The list of these controllers includes the

+ Replication Manager
+ ReplicaSet, DaemonSet, and Job controllers
+ Deployment controller
+ StatefulSet controller
+ Node controller
+ Service controller
+ Endpoints controller
+ Namespace controller
+ PersistentVolume controller
+ Others

Controllers do many different things, but they all watch the API server for changes
to resources and perform operations for each change.

In general, controllers run a reconciliation loop, which reconciles the actual state with
the desired state and writes the new actual states to the resource's `status` section.

### 11.1.7 What the Kubelet does

The Kubelet is the component responsible for everything running on a worker node. Its
initial job is to register the node it's running on by creating a Node resource in the
API server. Then it needs to continuously monitor the API server for Pods that
have been scheduled to the node, and start the pod's containers. It does this
by telling the configured container runtime to run a specific container image.

### 11.1.8 The role of the Kubernetes Service Proxy

Beside the Kubelet, every worker node also runs the kube-proxy, whose purpose is to
make sure clients can connect to the services you define through the
Kubernetes API. The kube-proxy makes sure connections to the service IP and port end
up at one of the pods backing that service. When a service is backed by more than one
pod, the proxy performs load balancing across those pods.

## 11.2 How controllers cooperate

Let's go over what happens when a Pod resource is created. You're going to create a
Deployment resource instead.

### 11.2.1 Understanding which components are involved

Even before you start the whole process, the controllers, the Scheduler, and the Kubelet
are watching the API server for changes to their respective resource types.

![k8s components watching API objects through the API server](https://s2.loli.net/2023/08/09/ezqxjhWnsw41OGc.png)

### 11.2.2 The chain of events

Imagine you prepared the YAML file containing the Deployment manifest and you're
about to submit it to Kubernetes through `kubectl`. `kubectl` sends the manifest to
API server in an HTTP POST request. The API server validates the Deployment specification,
stores it in etcd, and returns a response to `kubectl`. Now a chain of events starts to
unfold.

![The chain of events that unfolds when a Deployment resource is posted to the API server](https://s2.loli.net/2023/08/09/SYzaRhjtEdu4ymQ.png)

All API server clients watching the list of Deployment through the API server's watch
mechanism are notified of the newly created Deployment resource immediately after it's
created. And it will create a new ReplicaSet resource through API.

The newly created ReplicaSet is then picked up by the ReplicaSet controller, which watches
for creations, modifications, and deletions of ReplicaSet resources in the API server.
The controller then creates the Pod resources based on the pod template in the
ReplicaSet.

These newly created Pods are now stored in the etcd, but they each still lack one
important thing, they don't have an associated node yet. Their `nodeName`
attribute isn't set. The Scheduler watches for Pods like this, and when it encounters
one, chooses the best node for the Pod and assigns the Pod to the node.

With the Pod now scheduled to a specific node, the Kubelet on that node can finally
get to work. The Kubelet, watching for changes to Pods on the API server, sees a
new Pod scheduled to its node, so it inspects the Pod definition and instructs
Docker, or whatever container runtime it's using, to start the pod's containers.

### 11.2.3 Observing cluster events

Both the Control Plane components and the Kubelet emit events to the API server as
they perform these actions. They do this by creating Event resources.

## 11.3 Understanding what a running pod is

With the pod now running, let's look more closely at what a running pod even is.

When creating a pod, the pod would first creates some infrastructure containers.
The pause container is one whose sole purpose it to hold all these namespaces.

## 11.4 Inter-pod networking

Each pod gets its own unique IP address and can communicate with all other pods
through a flat, NAT-less network. The network is set up by the system administrator
or by a Container Network Interface (CNI) plugin.

### 11.4.1 What the network must be like

Kubernetes doesn't require you to use a specific networking technology, but it does
mandate that the pods can communicate with each other, regardless if they're running
on the same worker node or not.

When pod A connects to pod B, the source IP pod B sees must be the same IP that pod
A sees as its own. There should be not NAT performed.

### 11.4.2 Diving deeper into how networking works

A pod's network interface is set up in the infrastructure container. Let's see how
the interface is created and how it's connected to the interfaces in all the other
pods.

![Pods on a node are connected to the same bridge through veth](https://s2.loli.net/2023/08/10/PkhNirFXtWYxoCK.png)

Before the infrastructure container is started, a virtual Ethernet interface pair
is created for the container. One interface of the pair remains in the host's namespace,
whereas the other is moved into the container's network namespace and renamed `eth0`.

The interface in the host's network namespace is attached to a network bridge that the
container runtime is configured to use. The `eth0` interface in the container is
assigned an IP address from the bridge's address range. Anything that an application
running inside the container sends to the `eth0` network interface, comes out at the
other veth interface in the host's namespace and is sent to the bridge.

You have many ways to connect bridges on different nodes. This can be done with overlay
or underlay networks.

You know pod IP addresses must be unique across the whole cluster, so the bridges across
the nodes must use non-overlapping address ranges to prevent nodes on different nodes
from getting the same IP. Below, the bridge on node A is using the 10.1.1.0/24 IP range and
the bridge on node B is using 10.1.2.0/24, which ensures no IP address conflicts exist.

Also, below shows that to enable communication between pods across two nodes with plain
layer 3 networking, the node's physical network interface needs to be connected to
the bridge as well. Routing tables on node A need to be configured so all packets destined
for 10.1.2.0/24 are routed to node B.

![Different pods communication](https://s2.loli.net/2023/08/10/av9cUrfDX6KOL4o.png)

## 11.5 How services are implemented

### 11.5.1 Introducing the kube-proxy

Everything related to Services is handled by the kube-proxy process running on each node.
Initially, the kube-proxy was an actual proxy waiting for connections and for each
incoming connection, opening a new connection to one of the pods. This was called the
`userspace` proxy mode. Later, a better-performing `iptables` proxy mode replaced it.

### 11.5.2 How kube-proxy uses iptables

When a service is created in the API server, the virtual IP address is assigned to it
immediately. The API server notifies all kube-proxy agents running on the worker nodes
that a new Service has been created. Then each kube-proxy makes that service addressable
on the node it's running on. It does this by setting up a few `iptables` rules, which
make sure each packet destined for the service IP/port pair is intercepted and its
destination address modified.
