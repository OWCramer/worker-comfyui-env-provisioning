class ProvisionError(Exception):
    """Base class for all provisioning failures."""


class ConfigError(ProvisionError):
    """The environment variable configuration is invalid."""


class ResolutionError(ProvisionError):
    """A model URL could not be resolved to a downloadable file."""


class DownloadError(ProvisionError):
    """A model file could not be downloaded."""


class NodeInstallError(ProvisionError):
    """A custom node could not be installed."""
