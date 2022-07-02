# File System

The xv6 file system provides Unix-like files, descriptors, and pathnames,
and stores its data on a virtio disk for persistence. The file
system addresses several challenges:

+ The file system needs on-disk data structures to represent
the tree of named directories and files, to record the identifies of
the blocks that hold each file's content, and to record which
areas of the disk are free.
+ The file system must support *crash recovery*
+ Different processes may operate on the file system at the same
time, so the file-system code must coordinate to maintain invariants.
+ Accessing a disk is orders of magnitude slower than accessing memory,
so the file system must maintain an in-memory cache of popular blocks.

## Structure of the File System

The xv6 file system implementation is organized in seven layers shown
below. The disk layer reads and writes blocks on an virtio hard drive.
The buffer cache layer caches disk blocks and synchronizes access
to them, making sure
that only one kernel process at a time can modify the data stored in
any particular block. The logging layer allows higher layers to
wrap updates to several blocks in a *transaction*, and ensures that
the blocks are updated automatically in the face of crashes.
The inode layer provides individual files, each represented
as an *inode* with a unique o-number and some blocks holding
the file's data. The directory layer implements each directory
as a special kind of inode whose content is a sequence of directory
entries, each of which contains a file's name and i-number.
The pathname layer provides hierarchical path names and resolves
them with recursive lookup.

![Layers of the xv6 file system](https://s2.loli.net/2022/07/02/xhoeaAlIGMSVUJD.png)

Disk hardware traditionally presents the data on the disk as
a numbered sequence of 512-byte *blocks*. xv6 holds copies of
blocks in `buf.h`.

```c
struct buf {
  int valid;
  int disk;
  uint dev;
  uint blockno;
  struct sleep lock;
  uint refcnt;
  struct buf *prev;
  struct buf *next;
  uchar data[BSIZE];
}
```

xv6 divides the disk into several sections, as below shows. The
file system does not use block 0. Block 1 is called the *superblock*;
it contains metadata about the file system. Blocks starting at
2 hold the log. After the log are the inodes, with multiple inodes
per block. After those come bitmap blocks tracking which data blocks
are in use.
The superblock is filled in by a separate program, called `mkfs`,
which builds an initial file system.

![Structure of the xv6 file system](https://s2.loli.net/2022/07/02/iuYLZftrxhv8D65.png)

## Low Level

### Buffer Cache Layer

xv6 first defines a global `bcache` with a `head` node, and `head.next`
points to the most recently used.

```c
struct {
  struct spinlock lock;
  struct buf buf[NBUF];
  struct buf head;
}
```

xv6 uses `binit` to initialize the data structure and maintain LRU.
It uses the idea of deque, and makes the `head` as the sentinel
to achieve this data structure.

```c
void binit(void) {
  struct buf *b;

  initlock(&bcache.lock, "bcache");
  bcache.head.prev = &bcache.head;
  bcache.head.next = &bcache.head;
  for(b = bcache.buf; b < bcache.buf+NBUF; b++) {
    b->next = bcache.head.next;
    b->prev = &bcache.head;
    initsleeplock(&b->lock, "buffer");
    bcache.head.next->prev = b;
    bcache.head.next = b;
  }
}
```

xv6 provides `bread` interface to the upper level, it returns
a locked buf with the contents of the specified block.

```c
struct buf* bread(uint dev, uint blockno) {
  struct buf* b;
  b = bget(dev, blockno);
  if(!b->valid) {
    virtio_disk_rw(b, 0);
    b->valid = 1;
  }
  return b;
}
```

The actual operation is happen in the `bget`.

```c
static struct buf* bget(uint dev, uint blockno) {
  struct buf* b;
  acquire(&bcache.block);

  for(b = bcache.head.next; b != &bcache.head; b = b->next){
    if(b->dev == dev && b->blockno == blockno){
      b->refcnt++;
      release(&bcache.lock);
      acquiresleep(&b->lock);
      return b;
    }
  }

  for(b = bcache.head.prev; b != &bcache.head; b = b->prev){
    if(b->refcnt == 0) {
      b->dev = dev;
      b->blockno = blockno;
      b->valid = 0;
      b->refcnt = 1;
      release(&bcache.lock);
      acquiresleep(&b->lock);
      return b;
    }
  }
  panic("bget: no buffers");
}
```

`bget` first iterates the cache list to find whether the block
is already cached. If so, let reference plus one and do concurrency
operations. If not cached, recycle the LRU unused buffer.

xv6 also provides `bwrite` interface for upper level, it is easy.

```c
void
bwrite(struct buf *b) {
  if(!holdingsleep(&b->lock))
    panic("bwrite");
  virtio_disk_rw(b, 1);
}
```

At the end, xv6 defines `brelse` function to release a locked buffer,
and maintain the LRU.

```c

void brelse(struct buf *b) {
  if(!holdingsleep(&b->lock))
    panic("brelse");

  releasesleep(&b->lock);

  acquire(&bcache.lock);
  b->refcnt--;
  if (b->refcnt == 0) {
    // no one is waiting for it.
    b->next->prev = b->prev;
    b->prev->next = b->next;
    b->next = bcache.head.next;
    b->prev = &bcache.head;
    bcache.head.next->prev = b;
    bcache.head.next = b;
  }

  release(&bcache.lock);
}
```

### Logging Layer

An xv6 system call does not directly write the on-disk file system
data structures. Instead, it places a description of all the disk
writes it wishes to make in a *log* on the disk. Once the system
call has logged all of its writes, it writes a special *commit*
record to the disk indicating that the log contains a complete
operation. At that point the system call copies the writes to the
on-disk file system structures. After those writes have completed,
the system call erases the log on the disk.

The idea is simple. But the implementation is tricky. xv6 defines the `log` data structure.

```c
struct log {
  struct spinlock lock;
  int start;
  int size;
  int outstanding; // how many FS sys calls are executing.
  int committing;
  int dev;
  struct logheader lh;
}
```

`logheader` is used to keep track in memory of logged block before commit.

```c
struct logheader {
  int n;
  int block[LOGSIZE];
}
```

xv6 provides `begin_op` interface for calling at the start of
each FS system call.

```c
void begin_op(void) {
  acquire(&log.lock);
  while(1) {
    if(log.committing) {
      sleep(&log, &log.lock);
    } else if(log.lh.b + (log.outstanding+1)*MAXOPBLOCKS > LOGSIZE) {
      sleep(&log, &log.lock);
    } else {
      log.outstanding += 1;
      release(&log.lock);
      break;
    }
  }
}
```

`begin_op` is used to count. If the `log` is committing, we make
it sleep, if the space is not enough, we also make it sleep.

xv6 provides `end_op` interface for calling at the end of each
FS system call.

```c
void end_op(void) {
  int do_commit = 0;

  acquire(&log.lock);
  log.outstanding -= 1;

  if(log.committing)
    panic("log.committing");
  if(log.outstanding == 0) {
    do_commit = 1;
    log.committing = 1;
  } else {
    wakeup(&log);
  }
  release(&log.lock);

  if(do_commit) {
    commit();
    acquire(&log.lock);
    log.committing = 0;
    wakeup(&log);
    release(&log.lock);
  }
}
```

The process is simple, actually it's just concurrency operation. The
most important operation is `commit`.

```c
static void commit() {
  if(log.lh.n > 0) {
    write_log();
    write_head();
    install_trans(0);
    log.lh.n = 0;
    write_head();
  }
}
```

`write_log` copies modified blocks from cache to log.

```c
static void write_log(void) {
  int tail;

  for(tail = 0; tail < log.lh.n; tail++) {
    struct buf* to = bread(log.dev, log.start+tail+1);
    struct buf* from = bread(log.dev, log.lh.block[tail]);
    memmove(to->data, from->data, BSIZE);
    bwrite(to);
    brelse(from);
    brelse(to);
  }
}
```

`write_head` writes log header to disk. This is the most important thing
to be done to recover the file system. We need to record it.

```c
static void write_head(void) {
  struct buf *buf = bread(log.dev, log.start);
  struct logheader *hb = (struct logheader *) (buf->data);
  int i;
  hb->n = log.lh.n;
  for (i = 0; i < log.lh.n; i++) {
    hb->block[i] = log.lh.block[i];
  }
  bwrite(buf);
  brelse(buf);
}
```

Next, we look at `install_trans`, which just copies committed
blocks from log to their home location.

```c
static void install_trans(int recovering) {
  int tail;

  for (tail = 0; tail < log.lh.n; tail++) {
    struct buf *lbuf = bread(log.dev, log.start+tail+1); // read log block
    struct buf *dbuf = bread(log.dev, log.lh.block[tail]); // read dst
    memmove(dbuf->data, lbuf->data, BSIZE);  // copy block to dst
    bwrite(dbuf);  // write dst to disk
    if(recovering == 0)
      bunpin(dbuf);
    brelse(lbuf);
    brelse(dbuf);
  }
}
```

xv6 also defines `recover_from_log` to finish recovery.

```c
static void recover_from_log(void) {
  read_head();
  install_trans(1); // if committed, copy from log to disk
  log.lh.n = 0;
  write_head(); // clear the log
}
```

As you can see, the basic idea is simple. We need to know which blocks
the user want to writes, we need to write it all at once (transaction). So
we need to do a mapping.

## High Level

### Superblock and Bitmap

Now we comes to the high level part. How do we get the overall information
about the structure of the file system. xv6 uses `superblock`
to hold the information.

```c
struct superblock {
  uint magic;
  uint size;
  uint nblocks;
  uint ninodes;
  uint nlog;
  uint logstart;
  uint inodestart;
  uint bmapstart;
}
```

So for initializing the file system, we use `fsinit`.

```c
void fsinit(int dev) {
  readsb(dev, &sb);
  if(sb.magic != FSMAGIC)
    panic("invalid file system");
  initlog(dev, &sb);
}
```

`readsb` uses `bread` to read the block 1 of the disk, it is easy.

```c
static void readsb(int dev, struct superblock* sb) {
  struct buf* bp;
  bp = bread(dev, 1);
  memmove(sb, bp->data, sizeof(*sb));
  brelse(bp);
}
```

xv6 uses `balloc` to allocate a zeroed disk block. Well, just a bitmap.
The idea is simple. I omit detail here.

### Inode Layer

First, we need to create a inode table in memory to hold the information
about the on-disk inode.

```c
struct {
  struct spinlock lock;
  struct inode inode[NINODE];
} itable;
```

Let's look at the `struct inode` which defines in-memory inode:

```c
struct inode {
  uint dev;
  uint inum;
  int ref;
  struct sleeplock lock;
  int valid;
  short type;
  short major;
  short minor;
  short nlink;
  uint size;
  uint addrs[NDIRECT+1];
}
```

The `type` field distinguishes between files, directories, and
special files (devices). A type of zero indicates that an on-disk inode
is free. The `nlink` field counts the number of directory entries
that refer to this node, in order to recognize when on-disk inode and
its data blocks should be freed. The `size` field records the number
of bytes of content in the file. The `addrs` array records the block numbers
of the disk blocks holding the file's content.

Next, xv6 defines `iinit` to finish lock initialization.

```c
void iinit() {
  int i = 0;
  initlock(&itable.lock, "itable");
  for(int i = 0; i < NINODE; i++) {
    initsleeplock(&itable.inode[i].lock, "inode");
  }
}
```

xv6 also defines on-disk inode structure.

```c
struct dinode {
  short type;
  short major;
  short minor;
  short nlink;
  uint size;
  uint addrs[NDIRECT + 1];
}
```

However, there are too many codes to talk about. I don't think
this is a good idea to take notes. So I omit the detail.
