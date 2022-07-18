# Type

In order to define the type more concretely, enForwarderBot uses
`typing` module. The code can be viewed in `ehforwarderbot/types.py`.

We first look at the top code.

```python
if TYPE_CHECKING:
    from .chat import ChatMember
```

We first look at the definition of the `TYPE_CHECKING`. See
[Python official document](https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING).

> A special constant that is assumed to be `True` by 3rd party static type checkers. It is `False` at runtime.

So the type is an illusion made by Python. In runtime, we do
not care about the type. But when coding, type is an important thing.

And the code uses `NewType` helper to create distinct types.

```python
ReactionName = NewType('ReactionName', str)
ModuleID = NewType('ModuleID', str)
InstanceID = NewType('InstanceID', str)
ChatID = NewType('ChatID', str)
MessageID = NewType('MessageID', str)
ExtraCommandName = NewType('ExtraCommandName', str)
```

+ `ReactionName`: Canonical representation of a reaction, usually an emoji.
+ `ModuleID`: Module ID, including instance ID after `#` if available.
+ `InstanceID`: Instance ID of a module.
+ `ChatID`: Chat ID from slave channel or middleware, applicable to both chat and chat members.
+ `MessageID`: Message ID from slave channel or middleware.
+ `ExtraCommandName`: Command name of additional features.

And `Reactions` type is a mapping from `ReactionName` to `ChatMember` array.

```python
Reactions = Mapping[ReactionName, Collection['ChatMember']]
```

## Reference

+ [Custom Type Hints](https://ehforwarderbot.readthedocs.io/en/latest/API/types.html)

+ [types.py source code](https://github.com/ehForwarderBot/ehForwarderBot/blob/master/ehforwarderbot/types.py)
