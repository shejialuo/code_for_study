# Message Type

ehForwarderBot defines `MsgType` enum class. This part is easy.

```python
class MsgType(Enum):
    Text = "Text"
    Image = "Image"
    Voice = "Voice"
    Animation = "Animation"
    Video = "Video"
    File = "File"
    Location = "Location"
    Link = "Link"
    Sticker = "Sticker"
    Status = "Status"
    Unsupported = "Unsupported"
```

It's easy to understand.

## Reference

+ [Constants](https://ehforwarderbot.readthedocs.io/en/latest/API/constants.html#module-ehforwarderbot.constants)
+ [constants.py source code](https://github.com/ehForwarderBot/ehForwarderBot/blob/master/ehforwarderbot/constants.py)
