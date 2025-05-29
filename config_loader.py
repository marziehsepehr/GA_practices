import importlib.util

def load_config(config_path):
    """
    Dynamically load a configuration file and return its attributes as a dictionary.
    :param config_path: Path to the config.py file.
    :return: Dictionary containing the configuration variables.
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config
