import json


def get_tool_dict():
    """Placeholder for tool loading (can be extended if needed)"""
    return {}


def check_and_create_dir(path):
    """Check if directory exists, create if not"""
    import os
    if not os.path.exists(path):
        os.makedirs(path)
