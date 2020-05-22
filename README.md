# Synthetic Data Experiments with Bayesian Inference

Need to edit `matplotlib`'s `font_manager.py` to get multiprocessing to work correctly in the python code on Mac OS Catalina. Change:
```
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_get_font.cache_clear)
```

To:
```
if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_get_font.cache_clear)
```
