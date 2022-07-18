# Channel

There are two kinds of channel in ehForwarderBot.

+ master channel: The channel that directly interact with the User.
It is guaranteed to have one and only one master channel in
an EFB session.
+ slave channel: The channel that delivers messages to and from
their relative platform. There is at lease one slave channel in an EFB session.

So the code first defines an abstract class `Channel`.

```python
class Channel(ABC):
    channel_name: str = "Empty channel"
    channel_emoji: str = "�"
    channel_id: ModuleID = ModuleID("efb.empty_channel")
    instance_id : Optional[InstanceID] = None
    __version__ : str = 'undefined version'
```

The attribute `Channel` defined is easy to understand. I omit detail here

Now for the constructor.

```python
class Channel(ABC):
    def __init__(self, instance_id: InstanceID = None):
        if instance_id:
            self.instance_id = InstanceID(instance_id)
            self.channel_id = ModuleID(self.channel_id + "#" + instance_id)
```

The `Channel` class defines many abstract methods.

```py
@abstractmethod
def send_message(self, msg: 'Message') -> 'Message':
    raise NotImplementedError()
@abstractmethod
def poll(self):
    raise NotImplementedError()
@abstractmethod
def send_status(self, status: 'Status'):
    raise NotImplementedError()
@abstractmethod
def stop_polling(self):
    raise NotImplementedError()
@abstractmethod
def get_message_by_id(self, chat: 'Chat', msg_id: Message) -> Optional['Message']:
    raise NotImplementedError()
```

You could see the comments in the source which has a good description
about these functions and the specs.

Pay attention, the `send_status` method doesn't apply to the slave channel.

For master channel the definition is easy.

```python
class MasterChannel(Channel, ABC):

```

For slave channel, there are more attributes. First, it should
determine the message type it supports and the reactions it suggests
and some methods to get the chat information.

```python
class SlaveChannel(Channel, ABC):
    supported_message_types: Set[MsgType] = set()
    suggested_reactions: Optional[Sequence[ReactionName]] = None

    def get_extra_functions(self) -> Dict[ExtraCommandName, Callable]:
        methods = {}
        for mName in dir(self):
            m = getattr(self, mName)
            if callable(m) and getattr(m, "extra_fn", False):
                methods[ExtraCommandName(mName)] = m
        return methods
```

ehForwarderBot has provided `extra` in `utils.py` to set the
attribute of the provided function. So we use `get_extra_functions`
to get the information with the spec.

Again, there are some abstract methods.

```python
class SlaveChannel(Channel, ABC):
    @abstractmethod
    def get_chat_picture(self, chat: 'Chat') -> BinaryIO:
        raise NotImplementedError()
    @abstractmethod
    def get_chat(self, chat_uid: ChatID) -> 'Chat':
        raise NotImplementedError()
    @abstractmethod
    def get_chats(self) -> Collection['Chat']:
        raise NotImplementedError()
```

## Reference

+ [channel.py source code](https://github.com/ehForwarderBot/ehForwarderBot/blob/master/ehforwarderbot/channel.py)
+ [channel concept](https://ehforwarderbot.readthedocs.io/en/latest/guide/walkthrough.html#walk-through-how-efb-works)
