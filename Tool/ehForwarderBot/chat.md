# Chat

Chat is a very important concept in ehForwarderBot, there are three
kinds of chats:

1. Private chat: A conversation with a single person on the IM platform.
2. Group chat: a group that involves more than two members. A group
chat MUST provide a list of members that is involved in the conversation.
3. System chat: A chat that is a part of the system.

First, it defines the chat notification state.

```python
class ChatNotificationState(Enum):
    NONE = 0
    MENTIONS = 1
    ALL = -1
```

And it defines the base abstract class.

```python
class BaseChat(ABC):
    def __init__(self, *, channel: Optional[SlaveChannel] = None,
                 middleware: Optional[Middleware] = None,
                 module_name: str = "",
                 channel_emoji: str = "",
                 module_id: ModuleID = ModuleID(""),
                 name: str = "",
                 alias: Optional[str] = None,
                 uid: ChatID = ChatID(""),
                 id: ChatID = ChatID(""),
                 vendor_specific: Dict[str, Any] = None,
                 description: str = ""):
        if channel:
            if isinstance(channel, SlaveChannel):
                self.module_name = channel.channel_name
                self.channel_emoji = channel.channel_emoji
                self.module_id = channel.channel_id
            else:
                raise ValueError(
                    "channel value should be an SlaveChannel object"
                )
            elif isinstance(middleware, Middleware):
                self.module_id = middleware.middleware_id
                self.module_name = middleware.middleware_name
                self.channel_emoji = ""
            else:
                self.module_name = module_name
                self.channel_emoji = channel_emoji
                self.module_id = module_id

            self.name: str = name
            self.alias: Optional[str] = alias
            if id:
                 warnings.warn("`id` argument is deprecated, use `uid` instead.", DeprecationWarning)
            else:
                self.uid = uid
            self.vendor_specific: Dict[str, Any] = vendor_specific if vendor_specific is not None else dict()
            self.description: str = description

            @property
            def id(self) -> ChatID:
                warnings.warn("`id` property of a chat/member is deprecated, use `uid` instead.", DeprecationWarning)
                return self.uid

            @id.setter
            def id(self, val: ChatID):
                warnings.warn("`id` property of a chat/member is deprecated, use `uid` instead.", DeprecationWarning)
                self.uid = val

            @property
            def display_name(self) -> str:
                return self.alias or self.name

            @property
            def long_name(self) -> str:
                if self.alias:
                    return "{0} ({1})".format(self.alias, self.name)
                else:
                    return self.name

            def copy(self: _BaseChatSelf) -> _BaseChatSelf:
                return copy.copy(self)
            ...
```

You may wonder why uses `*`, it disables positional arguments.
This function is easy to understand.

And the latter code is used this abstract code. I omit detail here.
