import warnings

# langchain-community prints a deprecation notice on import; hide it to keep logs clean.
warnings.filterwarnings("ignore", message=".*langchain-community.*")
