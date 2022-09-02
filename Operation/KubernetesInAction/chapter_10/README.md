# Chapter 10 StatefulSets: deploying replicated stateful applications

## 10.1 Replicating stateful pods

ReplicaSets create multiple pod replicas from a single pod template. These replicas
don't differ from each other, apart from their name and IP address.
If the pod template includes a volume, which refers to a specific
PersistentVolumeClaim, all replicas of the ReplicaSet will use
the exact same PersistentVolumeClaim and therefore the same PersistentVolume
bound by the claim.

Thus you can't use a ReplicaSet to run a distributed data store,
where each instance needs its own separate storage.

### 10.1.1 Running multiple replicas with separate storage for each

#### Creating pods manually

You could create pods manually and have each of them use its own
PersistentVolumeClaim, but because no ReplicaSet looks after them,
you'd need to manage them manually.

#### Using one replica per pod instance

Instead of creating pods directly, you could create multiple
ReplicaSets—one for each pod with each ReplicaSet's desired replica
count set to one, and each ReplicaSet's pod template referencing a
dedicated PersistentVolumeClaim.

### 10.1.2 Providing a stable identity for each pod

In addition to storage, certain clustered applications also require
that each instance has a long-lived stable identity. Pods can be
killed from time to time and replaced with new ones.

## 10.2 Understanding StatefulSets

Instead of using a ReplicaSet to run these types of pods, you create
a StatefulSet resource, which is specifically tailored to applications
where instances of the application must be treated as non-fungible
individuals, with each one having a stable name and state.

### 10.2.1 Comparing StatefulSets with ReplicaSets

We tend to treat our app instance as pets, where we give each instance
a name and take care of each instance individually. But it's usually
better to treat instances as cattle and not pay special attention to
each individual instance. This makes it easy to replace unhealthy
instances without giving it a second thought.

Instead of stateless app, it doesn't matter if an instances dies.
You can create a new instance and people won't notice the difference.

On the other hand, with stateful apps, an app instance is more
like a pet. When a pet dies, you can't go buy a new one and except
people not to notice. To replace a lost pet, you need to find a new
one that looks and behaves exactly like the old one.

### 10.2.2 Providing a stable network identity

Each pod created by a StatefulSet is assigned an ordinal index,
which is then used to derive the pod's name and hostname, and
to attach stable storage to the pod.

![Pods created by a StatefulSet](https://s2.loli.net/2022/09/02/rTm9sIkV8ygXtSP.png)

#### Introducing the governing service

With stateful pods, you usually want to operate on a specific pod
from the group, because they differ from each other.

For this reason, a StatefulSet requires you to create a corresponding
governing headless Service that's used to provide the actual network identity
to each pod. Through this Service, each pod gets its own DNS entry, so
it peers and possibly other clients in the cluster can address the pod
by its hostname.

For example, if the governing Service belongs to the `default` namespace
and is called `foo`, and one of the pods is called `A-0`,
you can read the pod through its fully qualified domain name,
which is `a-0.foo.default.svc.cluster.local`.

#### Replacing lost pets

When a pod instance managed by a StatefulSet disappears, the
StatefulSet makes sure it's replaced with a new instance. But in contrast
to ReplicaSets, the replacement pod gets the same name and hostname
as the pod has disappeared.

![A StatefulSet replaces a lost pod with a new one with the same identity](https://s2.loli.net/2022/09/02/emr9TF7Z6zScqI4.png)

#### Scaling a StatefulSet

Scaling the StatefulSet creates a new pod instance with the next
unused ordinal index. The nice thing about scaling down a StatefulSet
is the fact that you always know what pod will be removed. Again,
this is also in contrast to scaling down a ReplicaSet, where you
have no idea what instance will be deleted, and you can't even specify
which one you want removed first. Scaling down a StatefulSet always
removes the instances with the highest ordinal index first.

### 10.2.3 Providing stable dedicated storage to each stateful instance

#### Teaming up pod templates with volume claim templates

The StatefulSet has to create the PersistentVolumeClaims as well,
the same way it's creating the pods. For this reason, a StatefulSet can also
have one or more volume claim templates, which enable to stamp out
PersistentVolumeClaims along with each pod instance.

#### Understanding the creation and deletion of PersistentVolumeClaims

Scaling up a StatefulSet by one creates two or more API objects. Scaling
down, deletes only the pod, leaving the claims along.
