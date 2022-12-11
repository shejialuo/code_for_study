# Chapter 1 Introducing the Istio service mesh

*Service mesh* is a relatively recent term used to describe a decentralized
application-networking infrastructure that allows application to be secure,
resilient, observable, and controllable. It describes an architecture made up
of a data plane that uses application-layer proxies to manage network traffic
on behalf of an application and a control plane to manage proxies.

## 1.1 Challenges of going faster

The following things must be addressed when moving to a services-oriented
architecture:

+ Keeping faults from jumping isolation boundaries
+ Building applications/services capable of responding to changes in their environment
+ Building systems capable of running in partially failed conditions
+ Understanding what's happening to the overall system as it constantly changes and evolves
+ Inability to control the runtime behaviors of the system
+ Implementing strong security as the attack surface grows
+ Lowering the risk of making changes to the system
+ Enforcing policies about who or what can use system components, and when

### 1.1.1 Our cloud infrastructure is not reliable

Let's take a very simple example. Let's say a `Preference` service in charge of managing
customer preferences and ends up making calls to a `Customer` service. However, things would
get into trouble:

+ The `Customer` service is overloaded and running slowly.
+ The `Customer` service has a bug.
+ The network has firewalls that are slowing the traffic.
+ The network is congested and is slowing traffic.
+ The network experienced some failed hardware and is rerouting traffic.
+ The network card on the `Customer` service hardware is experiencing failures.

### 1.1.2 Making service interactions resilient

The `Preference` service can try a few things. It can retry the request in a scenario where
things are overloaded, that might just add to the downstream issues. If it does retry the
request, it cannot be sure that previous attempts didn't succeed. It can time out the request
after some threshold and throw an error. It can also retry to a different instance of the
`Customer` service, may be in a different availability zone.

Some patterns have evolved to help mitigate these types of scenarios and help make applications
more resilient to unplanned, unexpected failures:

+ *Client-side load balancing*: Give the client the list of possible endpoints, and let it decide
which one to call.
+ *Service discovery*: A mechanism for finding the periodically updated list of healthy endpoints
for a particular logical service.
+ *Circuit breaking*: Shed load for a period of time to a service that appears to be misbehaving.
+ *Bulkheading*: Limit client resource usage with explicit thresholds when making calls to a service.
+ *Timeouts*: Enforce time limitations on requests, sockets, liveness, and so on when making
calls to a service.
+ *Retries*: Retry a failed request.
+ *Retry budgets*: Apply constraints to retries.
+ *Deadlines*: Give requests context about how long a response may still be useful.

## 1.2 Solving these challenges with application libraries

There are some drawbacks with application libraries:

+ Expected assumptions of any application.
+ Introduce a new language or framework to implement a service.

## 1.3 Pushing these concerns to the infrastructure

Using a *proxy* is a way to move these horizontal concerns into the infrastructure. A proxy is
an intermediate infrastructure component that can handle connections and redirect them to appropriate
backends. We use proxies all the time to handle network traffic, enforce security, and load balance
work to back-end servers.

## 1.4 What's a service mesh

A *service mesh* is a distributed application infrastructure that is responsible for handling
network traffic on behalf of the application in a transparent, out-of-process manner. The below
shows how service proxies form the *data plane* through which all traffic is handled and observed.
The data plane is responsible for establishing, securing, and controlling the traffic through the
mesh. The data plane behavior is configured by the *control plane*. The control plane is the brains
of the mesh and exposes an API for operators to manipulate network behaviors.

![A service mesh architecture with data plane and control plane](https://s2.loli.net/2022/12/11/X8YbTP9n7B2N5eO.png)
