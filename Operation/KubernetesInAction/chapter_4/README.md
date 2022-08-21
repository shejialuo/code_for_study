# Chapter 4 Replication and other controllers

In real-world use cases, you want to your deployments to stay up and
running automatically and remain healthy without any manual intervention.
To do this, you almost never create pods directly. Instead, you create
other types of resources, which then create and manage the actual pods.

## 4.1 Keeping pods healthy

### 4.1.1 Introducing liveness probes

Kubernetes can check if a container is still alive through *liveness probes*.
You can specify a liveness probe for each container in the pod's specification.
Kubernetes will periodically execute the probe and restart the container
if the probe fails.

Kubernetes can probe a container using one of the three mechanisms:

+ An *HTTP GET* probe performs an HTTP GET request on the container's IP
address, a port and path you specify.
+ A *TCP Socket* probe tries to open a TCP connection to the specified
port of the container.
+ An *Exec* probe executes an arbitrary command inside the container
and checks the command's exit status code.

### 4.1.2 Creating an HTTP-based liveness probe

You'll create a new pod that includes an HTTP GET liveness probe.
The following listing shows the YAML for the pod.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kubia-liveness
spec:
  containers:
    - image: kubia
      name: kubia
      livenessProbe:
        httpGet:
          path: /
          port: 8080
      imagePullPolicy: Never
      ports:
       - containerPort: 8080
         protocol: TCP
```

## 4.2 Introducing ReplicationControllers

A ReplicationController is a Kubernetes resource that ensure its pods
are always kept running. If the pod disappears for any reason, the
ReplicationController notices the missing pod anc creates a replacement pod.

### 4.2.1 The operation of a ReplicationController

An ReplicationController constantly monitors the list of running pods and
make sure the actual number of pods of a "type" always matches the
desired number. If too few such pods are running, it creates new replicas
from a pod template. If too many such pods are running, it removes
the excess replicas.

#### Introducing the controller's reconciliation loop

![A ReplicationController's reconciliation loop](https://s2.loli.net/2022/08/21/tVpoubF4dDc2YPf.png)

#### Understanding the three parts of a ReplicationController

A ReplicationController has three essential parts:

+ A *label selector*, which determines what pods are in the ReplicationController's scope.
+ A *replica count*, which specifies the desired number of pods that should
be running.
+ A *pod template*, which is used when creating a new pod replicas.

![The three key parts of a ReplicationController](https://s2.loli.net/2022/08/21/kBR1qmUlFwiJZHe.png)

#### Understanding the effect of changing the controller's label selector or pod template

Changes to the label selector and the pod template have no effect
on existing pods. Changing the label selector makes the existing pods
fall out of the scope of the ReplicationController, so the controller
stops caring about them.

### 4.2.2 Creating a ReplicationController

You're going to create a YAML file called `kubia-rc.yaml` for your
ReplicationController, as shown below.

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: kubia
spec:
  replicas: 3
  selector:
    app: kubia
  template:
    metadata:
      labels:
        app: kubia
    spec:
      containers:
      - name: kubia
        image: luksa/kubia
        ports:
        - containerPort: 8080
```

To create the ReplicationController, use the `kubectl create` command.

```sh
kubectl create -f kubia-rc.yaml
```

### 4.2.3 Seeing the ReplicationController in action

Because no pods exist with the `app=kubia` label, the ReplicationController should
spin up three new pods from the pod template.

```sh
kubectl get pods
```

### 4.2.4 Changing the pod template

A ReplicationController's pod template can be modified at any time.
Changing the pod template is like replacing a cookie cutter with another one.
It will only affect the cookies you cut out afterward and will have no
effect on ones you've already cut. To modify the old pods, you'd need
to delete them and let the ReplicationController replace them with new
ones based on the new template.

![Changing pod template effect](https://s2.loli.net/2022/08/21/EVGHJboQfzA4Xud.png)

```sh
kubectl edit rc kubia
```

### 4.2.5 Horizontally scaling pods

```sh
kubectl scale rc kubia --replicas=10
kubectl scale rc kubia --replicas=3
```
All these commands is modify the `spec.replicas` fields of the
ReplicationController's definition.

### 4.2.6 Deleting a ReplicationController

When you delete a ReplicationController through `kubectl delete`,
the pods are also deleted. But because pods created by a ReplicationController
aren't an integral part of the ReplicationController, and are only
managed by ti, you can delete only the ReplicationController and
leaves the pods running.

```sh
kubectl delete rc kubia --cascade=false
```

## 4.3 Using ReplicaSets instead of ReplicationControllers

ReplicaSet is a new generation of ReplicationController and replaces it completely.

### 4.3.1 Comparing a ReplicaSet to a ReplicationController

A ReplicaSet behaves exactly like a ReplicationController, but it has
more expressive pod selectors.

### 4.3.2 Defining a ReplicaSet

```yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: kubia
spec:
  replicas: 3
  selector:
    matchLabels:
      app: kubia
  template:
    metadata:
      labels:
        app: kubia
    spec:
      containers:
      - name: kubia
        image: luksa/kubia
        ports:
        - containerPort: 8080
```

The only difference is in the selector. Instead of listing labels
the pods need to have directly under the `selector` property, you're
specifying them under `selector.matchLabels`.

### 4.3.3 Using the ReplicaSet's more expressive label selectors

You have used the simpler `matchLabels` selector in the first ReplicaSet
example to see that ReplicaSets are no different from Replication Controllers.
Now you'll rewrite the selector to use the more powerful `matchExpressions` property.

```yaml
selector:
  matchExpressions:
    - key: app
      operator: In
      values:
        - kubia
```

You can add additional expressions to the selector. As in this example,
each expression must contain a `key`, an `operator`, and possibly a list
of `values`. You'll see four valid operators:

+ `In`: `Label`'s value must match one of the specified `values`.
+ `NotIn`: `Label`'s value must not match any of the specified `values`.
+ `Exists`: Pod must include a label with the specified key.
When using this operator, you shouldn't specify the `values` field.
+ `DoesNotExist`: Pod must not include a label with the specified key.
The `values` property must not be specified.

If you specify multiple expressions, all those expressions
must evaluate to true for the selector to match a pod. If you
specify both `matchLabels` and `matchExpressions`, all the labels
must match and all the expressions must evaluate to true for the pod
to match the selector.

## 4.4 Running exactly one pod on each node with DaemonSets

Sometimes you want a pod to run on each and every node in the cluster.
Those cases include infrastructure-related pods that perform system-level
operations. For example, you'll want to run a log collector and
a resource monitor on every node. Another good example is Kubernetes' own
kube-proxy process, which needs to run on all nodes to make services work.

### 4.4.1 Using a DaemonSet to run a pod on every node

To run a pod on all cluster nodes, you create a DaemonSet object.
A DaemonSet makes sure it creates as many pods as there are nodes
and deploys each one on its own node. If a node goes down, the DaemonSet
doesn't cause the pod to be created elsewhere. But when a new node
is added to the cluster, the DaemonSet immediately deploys a new pod
instance to it.

### 4.4.2 Using a DaemonSet to run pods only on certain nodes

A DaemonSet deploys pods to all nodes in the cluster, unless you specify
that the pods should only run on a subset of all the nodes. This is done
by specifying the `nodeSelector` property in the pod template.

```yaml
apiVersion: apps/v1
kind: DaemonSet
spec:
  selector:
    matchLabels:
      app: monitor
  template:
    metadata:
      labels:
        app: monitor
    spec:
      nodeSelector:
      containers:
      - name: kubia
        image: kubia
```

## 4.5 Running pods that perform a single completable task

Up to now, we've only talked about pods that need to run continuously.
You'll have cases where you only want to run a task that terminates
after completing its work.

Kubernetes includes support for Job resource, which allows you to
run a pod whose container isn't restarted when the process running inside
finishes successfully. Once it dose, the pod is considered complete.

In the event of a node failure, the pods on that node that are managed
by a Job will be rescheduled to other nodes. In the event of a failure
of the process itself, the Job can be configured to either restart
the container or not.
