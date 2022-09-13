# Raft

This paper is too long and every part you should read carefully.

## Decomposition of the problem

Raft decomposes the consensus problem into three parts:

+ *Leader election*: a new leader must be chosen when an existing leader fails.
+ *Log replication*: the leader must accept log entries from clients
and replicate them across the cluster, forcing the other logs
to agree with its own
+ *Safety*:
  + *Election Safety*: at most one leader can be elected in a given term.
  + *Leader Append-Only*: a leader new overwrites or deletes entries in its log;
  it only appends new entries.
  + *Log Matching*: if two logs contain an entry with the same index
  and term, then the logs are identical in all entries up through
  the given index.
  + *Leader Completeness*: if a log entry is committed in a given
  term, then the logs are identical in all entries up through the given index.
  + *State Machine Safety*: if a server has applied a log entry at
  a given index to its state machine, no other server will ever apply
  a different log entry for the same index.

So we can have got an import idea here: *control is independent with data*.
Although in the implementation, it may not.

## Basic concept

For servers, there are three states for leader election:

+ *leader*
+ *follower*
+ *candidate*

In normal operation there is exactly one leader and all of the other
servers are followers. Followers are passive: they issue no requests.
The third state, candidate, is used to elect a new leader.

Raft divides time into *terms* of arbitrary length, as shown below.
Terms are numbered with consecutive integers. Each term begins
with an *election*, in which one or more candidates attempt to
become leader. If a candidate wins the election, then it serves as
leader for the rest of the term. In some situations an election
will result in a split vote. In this case the term will end with
no leader; a new term (with a new election) will begin shortly.
Raft ensures that there is at most one leader in a given term.

![Time is divide into terms](https://s2.loli.net/2022/09/12/9QhB5IJDLGyv4SP.png)

Different servers may observe the transitions between terms at
different times, and in some situations a server may not observe
an election or even entire terms. Terms act as a logical clock in
Raft, and they allow servers to detect obsolete information
such as stale leaders. Each server stores a *current term* number, which
increases monotonically over time. Current terms are exchanged
whenever servers communicate; if one server's current term is
smaller than the other's, then it updates its current term to
the larger value. If a candidate or leader discovers that its term
is out of data, it immediately reverts to follower state. If
a server receives a request with a stale term number, it rejects the request.

Raft servers communicate using RPCs, and the basic consensus
algorithm requires only two types of RPCs. RequestVote RPCs
are initiated by candidates during elections, and AppendEntries RPCs
are initiated by leaders to replicate log entries and to provide
a form of heartbeat.

## Leader election

Raft uses a heartbeat mechanism to trigger leader election. When
servers start up, they begin as followers. A server remains in
follower state as long as it receives valid RPCs from a leader
or candidate. Leaders send periodic heartbeats to all followers in
order to maintain their authority. If a follower receives no
communication over a period of time called the *election timeout*,
then it assumes there is no viable leader and begins an election
to choose a new leader.

To begin an election, a follower increments its current term
and transitions to candidate state. It then votes for itself and
issues RequestVote RPCs in parallel to each of the other servers
in the cluster. A candidate continues in this state until one
of the three things happens:

1. It wins the election.
2. Another server establishes itself as leader.
3. A period time goes by with no winner.

A candidate wins an election if it receives votes from a majority
of the servers in the full cluster for the same term. Each server
will vote for at most one candidate in a given term, on a FCFS basis.
The majority rule ensures that at most one candidate can win the election
for a particular term. Once a candidate wins an election, it
becomes leader. It then sends heartbeat messages to all of the
other servers to establish its authority and prevent new elections.

While waiting for votes, a candidate may receive an AppendEntries RPC
from another server claiming to be leader. If the leader's term is
at least as large as the candidate's current term, then the candidate
recognizes the leader as legitimate and returns to follower state.
If the term in the RPC is smaller than the candidate's current
term, then the candidate rejects the RPC and continues in candidate
state.

The third possible outcome is that a candidate neither wins
nor loses the election: if many followers become candidates at
the same time, votes could be split so that no candidate obtains
a majority. When this happens, each candidate will time out and
start a new election by incrementing its term and initiating another
round of RequestVote RPCs. However, without extra measures split
votes could repeat indefinitely.

Raft uses randomized election timeouts to ensure that split votes
are rare and that they are resolved quickly. To prevent split
votes in the first place, election timeouts are chosen randomly from
a fixed interval.

## Log replication

Once a leader has been elected, it begins servicing client requests.
Each client request contains a command to be executed by the replicated
state machines. The leader appends the command to its log as a
new entry, then issues AppendEntries RPCs in parallel to each of
the other servers to replicate the entry. When the entry has
been safely replicated, the leader applies the entry to its state
machine and returns the result of that execution to the client.
If followers crash or run slowly, or if network packets are lost,
the leader retries AppendEntries RPCs indefinitely until all
followers *eventually store all log entries*.

Logs are organized as shown below. Each log entry stores a state machine
command along with the term number when the entry was received by the leader.
The term numbers in log entries are used to detect inconsistencies
between log. Each log entry also has an integer index identifying its
position in the log.

![Logs structure](https://s2.loli.net/2022/09/12/MPeYuHsZICgwLkR.png)

The leader decides when it is safe to apply a log entry to the
state machines; such an entry is called *committed*. Raft guarantees
that committed entries are durable and will eventually be executed
by all of the available state machines. A log entry is committed
once the leader that created the entry has replicated it on a
*majority* of the servers. The leader keeps track of the highest index
it knows to be committed, and it includes that index in future
AppendEntries RPCs so that the other servers eventually find out.
Once a follower learns that a log entry is committed, it applies the
entry to its local state machine.

Thus we constitute the Log Matching Property:

+ If two entries in different logs have the same index and term,
then they store the same command.
+ If two entries in different logs have the same index and term,
the the logs are identical in all preceding entries.

However, leader crashes can leave the logs inconsistent. These
inconsistencies can compound over a series of leader and follower
crashes. Below illustrates the ways in which followers' logs
may differ from that of a new leader. A follower may be missing entries
that are present on the leader, it may have extra entries that
are not present on the leader, or both. Missing and extraneous
entries in a log may span multiple terms.

![Possible inconsistency scenarios](https://s2.loli.net/2022/09/12/TPFUOX4c7zkfKbH.png)

In Raft, the leader handles inconsistencies by forcing the followers'
logs to duplicate its own. To bring a follower's log into consistency
with its own, the leader must find the latest log entry where
the two logs agree, delete any entries in the follower's log
after that point, and send the follower all of the leader's
entries after that point. All of these actions happen in response
to the consistency check performed by AppendEntries RPCs. The
leader maintains a *nextIndex* for each follower, which is the index
oft the next log entry the leader will send to that follower.

When a leader first comes to power, it initializes all nextIndex
values to the index just after the last one in its log. If a follower's
log is inconsistent with the leader's, the AppendEntries
consistency check will fail in the next AppendEntries RPC. After
a rejection, the leader decrements nextIndex and retries the
AppendEntries RPC. Eventually nextIndex will reach a point where
the leader and follower logs match. When this happens, AppendEntries
will succeed, which removes any conflicting entries in the follower's log
and appends entries from the leader's log.

With this mechanism, a leader does not need to take any special
actions to restore log consistency when it comes to power. It just
begins normal operation, and the logs automatically converge in response
to failures of the AppendEntries consistency check. A leader never
overwrites or deletes entries in its own log (the leader Append-Only Property).

## Safety

Now we come to the most important part, this part will make us
understand why Raft is correct. And how to make sure that we are correct.

### Election restriction

Raft uses an approach where it guarantees that all the committed
entries from previous terms are present on each new leader
from the moment of its election, without the need to transfer those
entries to the leader. This means that log entries only flow in
one direction, from leaders to followers, and leaders never overwrite
existing entries in their logs.

Raft uses the voting process to prevent a candidate from wining an
election unless its log contains all committed entries. A candidate
must contact a majority of the cluster in order to be elected,
which means that every committed entry must be present in
at least one of those servers. If the candidate's log is at least
as up-to-date as any other log in that majority, then it will hold
all the committed entries. The RequestVote RPC implements this
restriction: the RPC includes information about the candidate's log,
and the voter denies its vote if its own log is more up-to-date than
that of the candidate.

Raft determines which of two logs is more up-to-date by comparing the
index and term of the last entries in the logs. If the logs have
last entries with different terms, the the log with the later term
is more up-to-date.

### Committing entries from previous terms

A leader knows that an entry from its current term is committed once that
entry is stored on a majority of the servers. If a leader crashes
before committing an entry, future leaders will attempt to
finish replicating the entry. However, a leader cannot immediately
conclude that an entry from a previous term is committed
once it is stored on a majority of servers.

Raft never commits log entries from previous terms by counting replicas.
Only log entries from the leader's current term are committed by
counting replicas; once an entry from the current term has been
committed in this way, then all prior entries are committed indirectly
because of the Log Matching Property.

However, I omit the proof for Leader Completeness Property.

## Timing and availability

Leader election is the aspect of Raft where timing is most critical.
Raft will be able to elect and maintain a steady leader as long
as the system satisfies the following *timing requirement*:

$$
broadcastTime \ll electionTimeout \ll MTBF
$$
