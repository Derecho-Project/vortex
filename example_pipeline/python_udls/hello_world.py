def invoke(data) -> None:
    """prints hello world + the data sent into this module"""
    print(f"Hello, World! {data}")

    if isinstance(data, list):
        return [None] * len(data)

    return None