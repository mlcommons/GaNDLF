from GANDLF.cli import copyrightMessage


def append_copyright_to_help(command_func):
    if command_func.__doc__ is None:
        command_func.__doc__ = copyrightMessage
    else:
        command_func.__doc__ += '\n\n' + copyrightMessage
    return command_func
