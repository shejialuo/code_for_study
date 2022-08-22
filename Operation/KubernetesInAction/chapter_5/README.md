# Chapter 5 Services: enabling clients to discover and talk to pods

Pods need a way of finding other pods if they want to consume the services
they provide. Unlike in the non-kubernetes world, where a sysadmin would
configure each client app by specifying the exact IP address or hostname
of the server providing the service in the client's configuration files,
doing the same in Kubernetes wouldn't work, because

+ *Pods are ephemeral*.
+ *Kubernetes assigns an IP address* to a pod after the pod has been
scheduled to a node and before it's started.
+ *Horizontal scaling means multiple pods may provide the same service*.

## 5.1 Introducing services

A Kubernetes Service is a resource you create to make a single, constant point
of entry to a group of pods providing the same service.
Each service has an IP address and port that never change while the
service exists. Clients can open connections to that IP and port, and
those connections are then routed to one of the pods backing that service.

### 5.1.1 Creating services

#### Creating a service through kubeclt expose

The easiest way to create a service is through `kubectl expose`.

#### Creating a service through a YAML descriptor

```yaml
apiVersion: v1
kind: Service
metadata:
  name: kubia
spec:
  ports:
  - port: 80
    targetPort: 8080
  selector:
    app: kubia
```

You're defining a service called `kubia`, which will accept connections on
port 80 and route each connect to port 8080 of the pods matching the
`app=kubia` label selector

#### Examining your new service

```sh
kubectl get svc
```

#### Testing your service from within the cluster

You can send requests to your service from with the cluster in a few ways:

+ The obvious way is to create a pod that will send the request
to the service's cluster IP and log the response.
+ You can `ssh` into one of the Kubernetes nodes and use the `curl`.
+ You can execute the `curl` inside one of your existing pods through
the `kubectl exec`.

#### Configuring session affinity on the service

If you execute the same command a few more times, you should hit a
different pod with every invocation, because the service proxy normally
forwards each connection to a randomly selected backing pod,
even if the connections are coming from the same client.

If you want to all requests made by a certain client to be redirected
to the same pod every time, you can set the service's `sessionAffinity`
property to `ClientIP`.

#### Exposing multiple ports in the same service

Your service exposes only a single port, but services can also
support multiple ports.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: kubia
spec:
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: https
    port: 443
    targetPort: 8443
  selector:
    app: kubia
```

#### Using named ports

In all these examples, you've referred to the target port by its
number, but you can also give a name to each pod's port and refer
to it by name in the service `spec`. This makes the service `spec` slightly
clearer, especially if the port numbers aren't well-known.

For example, suppose your pod defines names for its ports as
shown in the following listing.

```yaml
kind: Pod
spec:
  containers:
  - name: kubia
    ports:
    - name: http
      containerPort: 8080
    - name: https:
      containerPort: 8443
```

You can then refer to those ports by name in the service `spec`.

```yaml
apiVersion: v1
kind: Service
spec:
  ports:
  - name: http
    port: 80
    targetPort: http
  - name: https
    port: 443
    targetPort: https
```

### 5.1.2 Discovering services

How do the client pods know the IP and port of a service?
Kubernetes provides ways for client pods to discover a service's IP and port.

#### Discovering services through environment variables

When a pod is started, Kubernetes initializes a set of environment
variables pointing to each service that exists at that moment.
If you create the service before creating the client pods, processes
in those pods can get the IP address and port of the service by
inspecting their environment variables.

#### Discovering services through DNS

Each service gets a DNS entry in the internal DNS server, and client
pods that know the name of the service can access it through its
fully qualified domain name (FQDN)

#### Connecting to the service through its FQDN

You'll try to access the `kubia` service through its FQDN instead of
its IP. Again, you'll need to that inside an existing pod.

```sh
kubectl exec -it kubia-r5qd6 -- bash
curl http://kubia.default.svc.cluster.local
curl http://kubia.default
curl http://kubia
```

## 5.2 Connecting to services living outside the cluster

Up to now, we've talked about services backed by one or more pods running
inside the cluster. But cases exist when you'd like to expose
external services through the Kubernetes service feature.
Instead of having the service redirect connections to pods in
the cluster, you want it to redirect to external IP and port.

### 5.2.1 Introducing service endpoints

```sh
kubectl get endpoints kubia
```

### 5.2.2 Manually configuring service endpoints

To create a service with manually managed endpoints, you need
to create both a Service and an Endpoints resource.

#### Creating a service without a selector

You'll first create the YAML for the service itself, as shown below.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: external-service
spec:
  ports:
  - port: 80
```

#### Creating an endpoints resource for a service without a selector

Endpoints are a separate resource and not an attribute of a service.

```yaml
apiVersion: v1
kind: Endpoints
metadata:
  name: external-service
subsets:
  - addresses:
    - ip: 11.11.11.11
    - ip: 22.22.22.22
    ports:
    - port: 80
```

The Endpoints object needs to have the same name as the service and
contain the list of target IP addresses and ports for the service.

![Pods consuming a service with two external endpoints](https://s2.loli.net/2022/08/22/ZNCgvYa79HPsjdM.png)

### 5.2.3 Creating an alias for an external service

Instead of exposing an external service by manually configuring the
service's Endpoints, a simpler method allows you to refer to an
external service by its fully qualified domain name(FQDN).

```yaml
apiVersion: v1
kind: Service
metadata:
  name: external-service
spec:
  type: ExternalName
  externalName: someapi.somecompany.com
  ports:
  - port: 80
```

After the service is created, pods can connect to the external
service through the `external-service` instead of using the service's
actual FQDN.
This hides the actual service name and its location from pods
consuming the service.

## 5.3 Exposing services to external clients

You'll also want to expose certain services to the outside, so
external clients can access them, as depicted below.

![Exposing a service to external clients](https://s2.loli.net/2022/08/22/BS9zgxn8k1w2q6G.png)

### 5.3.1 Using a NodePort service

The first method of exposing a set of pods to external clients
is by creating a service and settings its type to `NodePort`.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: kubia-nodeport
spec:
  type: NodePort
  ports:
  - port: 80
    targetPort: 8080
    nodePort: 30123
  selector:
    app: kubia
```

You set the type to `NodePort` and specify the node port this service
should be bound to across all cluster nodes. Specifying the port
isn't mandatory; Kubernetes will choose a random port if you omit it.

Below shows your service exposed on port 30123 of both your cluster nodes.

![An external client connecting to a NodePort](https://s2.loli.net/2022/08/22/AMmWhRCFpGqsozb.png)

### 5.3.2 Exposing a service through an external load balancer

```yaml
apiVersion: v1
kind: Service
metadata:
  name: kubia-loadbalancer
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8080
  selector:
    app: kubia
```

See blow to see how HTTP requests are delivered to the pod.
External clients connect to port 80 of the load balancer and get
routed to the implicitly assigned node port on one of the nodes.

A `LoadBalancer`-type service is a `NodePort` service with an additional
infrastructure-provided load balancer.

## 5.4 Exposing services externally through an Ingress resource

One important reason is that each `LoadBalancer` service requires its
own load balancer with its own public IP address, whereas an Ingress
only requires one, even when providing access to dozens of services.
When a client sends an HTTP request to the Ingress, the host and path
in the request determine which service the request is forwarded to, as shown below.

![Multiple services can be exposed through a single ingress](https://s2.loli.net/2022/08/22/BqgtikrUuhCbc7E.png)

## 5.5 Signaling when a pod is ready to accept connections

The pod may need time to load either configuration or data, or it
may need to perform a warm-up procedure to prevent the first user
request from taking too long and affect the user experience. In
such cases you don't want the pod to start receiving request immediately,
especially when the already-running instances can process requests
properly and quickly. It make sense to not forward requests to a pod
that's in the process of starting up until it's fully ready.

Similar to liveness probes, Kubernetes allows you to also define
a readiness probe for your pod.

The readiness probe is invoked periodically and determines whether
the specific pod should receive client requests or not. When a
container's readiness probe returns success, it's signaling that
the container is ready to accept requests.

Like liveness probes, three types of readiness probes exist:

+ An *Exec* probe.
+ An *HTTP Get* Probe.
+ A *Tcp Socket* probe.

When a container is started, Kubernetes can be configured to wait
for a configurable amount of time to pass before performing the first
readiness check. After that, it invokes the probe periodically and
acts based on the result of the readiness probe. If a pod reports that
it's not ready, it's removed from the service. If the pod then
becomes ready again, it's re-added.

Unlike liveness probes, if a container fails the readiness check,
it won't be killed or restarted.

As you can see in below, if a pod's readiness probe fails, the pod
is removed from the Endpoints object.

![A pod whose readiness probe is removed](https://s2.loli.net/2022/08/22/k3gO4uAa1XeDxTr.png)

## 5.6 Using a headless service for discovering individual pods

If you tell Kubernetes you don't need a cluster IP for your service,
the DNS server will return the pod IPs instead of the single service IP.
