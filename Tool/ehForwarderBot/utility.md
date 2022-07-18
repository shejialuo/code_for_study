# Utility

Utility provides some auxiliary functions to make the life easier.

First, ehForwarderBot providers slave channel some custom extra
functions.

```python
def extra(name: str, desc: str) -> Callable[..., Optional[str]]:
    def attr_dec(f):
        f.__setattr__("extra_fn", True)
        f.__setattr__("name", name)
        f.__setattr__("desc", desc)
        return f

    return attr_dec
```

And next uses `get_base_path` to get the configuration path basename.

```python
def get_base_path() -> Path:
    env_data_path = os.environ.get("EFB_DATA_PATH", None)
    if env_data_path:
        base_path = Path(env_data_path).resolve()
    else:
        base_path = Path.home() / ".ehforwarderbot"
    if not base_path.exists():
        base_path.mkdir(parents=True)
    return base_path
```

And also defines `get_data_path` to get the path for permanent storage
of a module.

```python
def get_data_path(module_id: ModuleID) -> path:
    profile = coordinator.profile
    data_path = get_base_path() / 'profiles' / profile / module_id
    if not data_path.exists():
        data_path.mkdir(parents=True)
    return data_path
```

At now, I have understood the configuration I have written for
setting up the EFB.

And for the other functions, it is easy to understand. I omit
detail here.

```python
def get_config_path(module_id: ModuleID = None, ext: str = 'yaml') -> Path:
    if module_id:
        config_path = get_data_path(module_id)
    else:
        profile = coordinator.profile
        config_path = get_base_path() / 'profiles' / profile
    if not config_path.exists():
        config_path.mkdir(parents=True)
    return config_path / "config.{}".format(ext)

def get_custom_modules_path() -> Path:
    channel_path = get_base_path() / "modules"
    if not channel_path.exists():
        channel_path.mkdir(parents=True)
    return channel_path
```

## Reference

+ [Utility](https://ehforwarderbot.readthedocs.io/en/latest/API/utils.html)
+ [utility.py source code](https://github.com/ehForwarderBot/ehForwarderBot/blob/master/ehforwarderbot/utils.py)
