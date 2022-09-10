# The Design of a Practical System for Fault-Tolerant Virtual Machines

## Introduction

A common approach to implementing fault-tolerant servers is the
primary/backup approach, where a backup server is always available
to take over if the primary server fails. One way of replicating the
state on the backup server is to ship changes to all state of
the primary, including CPU, memory, and I/O devices, to the backup
nearly continuously. However, the bandwidth needed to send this
state, particular changes in memory, can be very large.

A different method for replicating servers that can use much
less bandwidth is referred to as the state-machine approach.
The idea is to model the servers as deterministic state machines
that are kept in sync by starting them from the same initial state
and ensuring that they receive the same input requests in the
same order. Since most servers or services have some operations that
are not deterministic, extra coordination must be used to ensure that
a primary and backup are kept in sync.

A VM running on top of a hypervisor is an excellent platform for
implementing the state-machine approach. A VM can be considered
a well-defined state machine whose operations are the operations
of the machine being virtualized.

## Basic FT Design

Below shows the basic setup of our system for fault-tolerant VMs.
For a given VM for which we desire to provide fault tolerance
(the *primary* VM), we run a *backup* VM on a different physical server
that is kep in sync and executes identically to the primary virtual
machine, though with a small time lag. We says that the two VMs
are in *virtual-lockstep*.

![Basic FT Configuration](https://s2.loli.net/2022/09/07/QovRGeruz46kVD7.png)

### Deterministic Replay Implementation

There are events for VM:

+ Deterministic
+ Non-deterministic: virtual interrupts, reading the clock cycle
counter of the processor.

This presents three challenges for replicating execution of any
VM running any operating system and workload:

+ Correctly capturing all the input and non-determinism necessary
to ensure deterministic execution of a backup virtual machine
+ Correctly applying the inputs and non-determinism to the backup VM
+ Don't degrade performance

### FT Protocol

Instead of writing the log entries to disk, we send them to the backup VM
via the logging channel. The backup VM replays the entries in real time,
and hence executes identically to the primary VM. However, we must
augment the logging entries with a strict FT protocol.

+ **Output Requirement**: if the backup VM ever takes over after
a failure of the primary, the backup VM will continue executing in
way that is entirely consistent with all outputs that the primary
VM has sent to the external world.

The easiest way to enforce the Output Requirement is to create
a special log entry at each output operation. Then, the Output Requirement
may be enforced by this specific rule:

+ **Output Rule**: the primary VM may not send an output to the external
world, until the backup VM has received and acknowledged the log
entry associated with the operation producing the output.

Note that the Output Rule does not say anything about stopping the execution
of the primary VM. We need only delay the sending of the output, but the
VM itself can continue execution. Since operating systems do
non-blocking network and disk outputs with asynchronous interrupts
to indicate completion, the VM can easily continue execution and will
not necessarily be immediately affected by the delay in the output.

### Detecting and Responding to Failure

The primary and backup VMs must respond quickly if the other VM
appears to have failed. If the backup VM fails, the primary VM
will *go live*: leave recording mode and start executing normally.
If the primary VM fails, the backup VM should similarly *go live*.

There are many possible ways to attempt to detect failure of
the primary and backup VMw. VMware FT uses UDP heartbeating
between servers that are running fault-tolerant VMs to detect
when a server may have crashed.

To avoid split-drain problems, we make use of the shared storage
that stores the virtual disks of the VM. When either a primary or backup
VM wants to go live, it executes an atomic test-and-set operation
on the shared storage. If the operation succeeds, the VM is allowed
to go live. If the operation fails, then the other VM must have already
gone live, so the current VM actually halts itself.
