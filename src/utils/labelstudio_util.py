import os

def convert_labelstudio_image_path(path):
    """
    Convert image path from Label Studio JSON to a correct absolute path.
    
    :param path: The original path from Label Studio JSON.
    :return: The transformed absolute path.
    :raises FileNotFoundError: If the transformed path does not exist.
    """
    # Define the prefix to replace
    prefix_to_replace = "/data/local-files/?d="
    
    # Get the home directory from the environment variable
    home_dir = os.environ.get("HOME")
    
    if not home_dir:
        raise EnvironmentError("HOME environment variable is not set.")
    
    # Replace the prefix with the home directory path
    if path.startswith(prefix_to_replace):
        relative_path = path[len(prefix_to_replace):]
        absolute_path = os.path.join(home_dir, relative_path)
        
        # Check if the path exists
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"The path does not exist: {absolute_path}")