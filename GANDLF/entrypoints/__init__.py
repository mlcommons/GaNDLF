from GANDLF.cli import copyrightMessage


def append_copyright_to_help(command_func):
    command_func.__doc__ = (
        copyrightMessage
        if command_func.__doc__ is None
        else (command_func.__doc__ + "\n\n" + copyrightMessage)
    )
    return command_func
