# Chapter 3 Pods: running containers in Kubernetes

## 3.1 Introducing Pods

Instead of deploying containers individually, you always deploy and
operate on a pod of containers. We're not implying that a pod always
includes more than one container, however, it's common for pods
to contain only a single container. The key thing about pods
is that when a pod does contain multiple containers, all of them
are always run on a single worker node.

![All containers of a pod run on the same node](https://s2.loli.net/2022/08/21/gynGKmz2lA4pft7.png)

### 3.1.1 Understanding why we need pods

You need to run each process in its own container. That's how
Docker and Kubernetes are meant to be used.

### 3.1.2 Understanding pods

A pod of containers allows you to run closely related processes together
and provide them with almost the same environment as if they were
all running in a single container, while keeping them somewhat isolated.

#### Understanding the partial isolation between containers of the same pod

All containers of a pod run under the same Network and UTS namespaces,
they all share the same hostname and network interfaces. Similarly,
all containers of a pod run under the same IPC namespace and can
communicate through IPC.

But when it comes to the filesystem, things are a little different.
By default, the filesystem of each container is fully isolated from other
containers. However,
it's possible to have them share file directories using a Kubernetes concept
called a *Volume*.

#### Introducing the flat inter-pod network

All pods in a Kubernetes cluster reside in a single flat, shared,
network-address space (shown below), which means every pod can access
every other pod at the other pod's IP address.

![Each pod gets routable IP address](https://s2.loli.net/2022/08/21/DbVxPlh3SfE8cNk.png)

## 3.2 Creating pods from YAML or JSON descriptors

### 3.2.1 Introducing the main parts of a pod definition

The pod definition consists of a few parts. First, there's the Kubernetes API
version used in the YAML and the type of resource the YMAL is describing.
Then, three important sections are found in almost all Kubernetes resources:

+ *Metadata* includes the name, namespace, labels, and other information about the pod.
+ *Spec* contains the actual description of the pod's contents, such as
the pod's containers, volumes, and other data.
+ *Status* contains the current information about the running pod.

### 3.2.2 Creating a simple YAML descriptor for a pod

You're going to create a file called `kubia-manual.yaml`.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kubia-manual
spec:
  containers:
    - image: kubia
      name: kubia
      imagePullPolicy: Never
      ports:
       - containerPort: 8080
         protocol: TCP
```

### 3.2.3 Using kubectl create to create the pod

To create the pod from your YAML file, use the `kubectl create` command.


```sh
kubectl create -f kubia-manual.yaml
```

### 3.2.4 Viewing application logs

To see your pod's log you run the following command on your local machine.

```sh
kubectl logs kubia-manual
```

If your pod includes multiple containers, you have to explicit specify
the container name by including the `-c <container name>` option.

### 3.2.5 Sending requests to the pod

Kubernetes allows you to configure port forwarding to the pod. This
is done through `kubectl port-forward` command.

```sh
kubectl port-forward kubia-manual 8080:8080
```

## 3.3 Organizing pods with labels

Organize pods and all other Kubernetes objects is done through *labels*.

### 3.3.1 Introducing labels

A label is an arbitrary key-value pair you attach to a resource, which
is then utilized when selecting resources using *label selectors*.

### 3.3.2 Specifying labels when creating a pod

Now, you'll see labels in action by creating a new pod with two labels.
Create a new file called `kubia-manual-with-labels.yaml` with
the contents of the following listing.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kubia-manual-v2
  labels:
    creation_method: manual
    env: prod
spec:
  containers:
    - image: kubia
      name: kubia
      imagePullPolicy: Never
      ports:
       - containerPort: 8080
         protocol: TCP
```

```sh
kubectl create -f kubia-manual-with-labels.yaml
```

The `kubectl get pods` command doesn't list any labels by default,
but you can see them by using the `--show-labels` switch:

```sh
kubectl get pods --show-labels
```

Instead of listing all labels, if you're only interested in certain
labels, you can specify them with the `-L` switch and have each displayed
in its own column.

```sh
kubectl get pods -L creation_method,env
```

### 3.3.3 Modifying labels of existing nodes

Labels can be added to and modified on existing pods.

```sh
kubectl label po kubia-manual creation_method=manual
```

You need to use the `--overwrite` option when changing existing labels.

```sh
kubectl label po kubia-manual-v2 env=debug --overwrite
```

## 3.4 Listing subsets of pods through label selectors

Labels go hand in hand with *label selectors*. Label selectors allow
you to select a subset of pods tagged with certain labels and perform
an operation on those pods. A label selector is a criterion, which
filters resources based on whether they include a certain label with
a certain value.

A label selector can select resources based on whether the resource

+ Contains a label with a certain key.
+ Contains a label with a certain key and value.
+ Contains a label with a certain key, but with a value not equal to the one you specify.

### 3.4.1 Listing pods using a label selector

Let's use label selectors on the pods you've created so for. To
see all pods you created manually, do the following:

```sh
kubectl get po -l creation_method=manual
```

To list all pods that include the `env` label, whatever its value is:

```sh
kubectl get po -l env
```

And those that don't have the `env` label:

```sh
kubectl get po -l '!env'
```

Similarly, you could also match pods with the following label selectors:

+ `creation_method!=manual`.
+ `env in (prod, devel)`
+ `env notin (prod,devel)`

### 3.4.2 Using multiple conditions in a label selector

A selector can also include multiple comma-separated criteria.

## 3.5 Using labels and selectors to constrain pod scheduling

All the pods you've created so far have been scheduled pretty much
randomly across your worker nodes. However, you'll want to have
at least a little say in where a pod should be scheduled.
You never want to say specifically what node a pod should be scheduled to,
because that would couple the application to the infrastructure,
whereas the whole idea of Kubernetes is hiding the actual infrastructure
from the apps that run on it. You should describe the node requirements and
then let Kubernetes select a node that matches those requirements.
This can be done through node labels and node label selectors.

### 3.5.1 Using labels for categorizing worker nodes

```sh
kubectl label node minikube gpu=true
```

### 3.5.2 Scheduling pods to specific nodes

Creating a file called `kubia-gpu.yaml` with the following contents.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kubia-cpu
spec:
  nodeSelector:
    gpu: "true"
  containers:
    - image: kubia
      name: kubia
      imagePullPolicy: Never
      ports:
       - containerPort: 8080
         protocol: TCP
```

```sh
kubectl create -f kubia-gpu.yaml
```

### 3.5.3 Scheduling to one specific node

Similarly, you could also schedule a pod to an exact node, because each
node also has a unique label with the key `kubernetes.io/hostname` and
value set to the actual hostname of the node.

## 3.6 Annotating pods

In addition to labels, pods and other objects can also contain *annotations*.
Annotations are also key-value pairs, which can hold much larger pieces of
information and are primarily meant to be used by tools.

A great use of annotations is adding descriptions for each pod or
other API object, so that everyone using the cluster can quickly
look up information about each individual object.

### 3.6.1 Adding and modifying annotations

Annotations can obviously be added to pods at creation time, the
same way labels can. The simplest way to add an annotation to an
existing object is through the `kubectl annotate` command.

```sh
kubectl annotate pod kubia-manual mycompany.com/someannotation="foo bar"
```

When different tools or libraries add annotations to objects, they
may accidentally override each other's annotations if they don't use
unique prefixes.

You can use `kubectl describe` to see the annotation you added:

```sh
kubectl describe pod kubia-manual
```

## 3.7 Using namespaces to group resources

Kubernetes namespaces provide a scope for objects names.

### 3.7.1 Discovering namespaces and their pods

First, let's list all namespaces in your cluster:

```sh
kubectl get ns
```

Up to this point, you've operated only in the `default` namespace.
When listing resources with the `kubectl get` command, you've never
specified the namespace explicitly, so `kubectl` always defaulted to
the `default` namespace.

However, we could look up for the other namespaces.

```sh
kubectl get po --namespace kube-system
```

### 3.7.2 Creating a namespace

A namespace is a Kubernetes resource like any other, so you can create it by
posting a YAML file to the Kubernetes API server.

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: custom-namespace
```

```sh
kubectl create -f custom-namespace.yaml
```

You can also create namespaces with dedicated `kubectl create namespace`
command:

```sh
kubectl create namespace custom-namespace
```

### 3.7.3 Managing objects in other namespaces

To create resources in the namespace you've created, either add a
`namespace: custom-namespace` entry to the `metadata` section, or
specify the namespace when creating the resource with the `kubectl create` command.

```sh
kubectl create -f kubia-manual.yaml -n custom-namespace
```

## 3.8 Stopping and removing pods

### 3.8.1 Deleting a pod by bane

```sh
kubectl delete po kubia-gpu
```

### 3.8.2 Deleting pods using label selectors

```sh
kubectl delete po -l creation_method=manual
```

### 3.8.3 Deleting pods by deleting the whole namespace

```sh
kubectl delete ns custom-namespace
```

### 3.8.4 Deleting all pods in a namespace, while keeping the namespace

```sh
kubectl delete po --all
```
