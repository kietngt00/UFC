from contextlib import contextmanager

@contextmanager
def lora_enabled(model, name):
    model.set_adapter(name)
    try:
        yield
    finally:
        model.disable_adapters()