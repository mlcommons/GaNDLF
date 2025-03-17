import logging
from typing import Optional, Union


from typing import Type
from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails


def generate_and_save_markdown(model: Type[BaseModel], file_path: str) -> None:
    schema = model.schema()
    markdown = []

    # Add title
    markdown.append(f"# {schema['title']}\n")

    # Add description if available
    if "description" in schema:
        markdown.append(f"{schema['description']}\n")

    # Add fields table
    markdown.append("## Parameters\n")
    markdown.append("| Field | Type | Description | Default |")
    markdown.append(
        "|----------------|----------------|-----------------------|------------------|"
    )

    for field_name, field_info in schema["properties"].items():
        # Extract field details
        field_type = field_info.get("type", "N/A")
        description = field_info.get("description", "N/A")
        default = field_info.get("default", "N/A")

        # Add row to the table
        markdown.append(
            f"| `{field_name}` | `{field_type}` | {description} | `{default}` |"
        )

    # Write to file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write("\n".join(markdown))


def initialize_key(
    parameters: dict, key: str, value: Optional[Union[str, float, list, dict]] = None
) -> dict:
    """
    This function initializes a key in the parameters dictionary to a value if it is absent.

    Args:
        parameters (dict): The parameter dictionary.
        key (str): The key to initialize.
        value (Optional[Union[str, float, list, dict]], optional): The value to initialize. Defaults to None.

    Returns:
        dict: The parameter dictionary.
    """
    if parameters is None:
        parameters = {}
    if key in parameters:
        if parameters[key] is not None:
            if isinstance(parameters[key], dict):
                # if key is present but not defined
                if len(parameters[key]) == 0:
                    parameters[key] = value
    else:
        parameters[key] = value  # if key is absent

    return parameters


# Define custom errors
CUSTOM_MESSAGES = {
    "literal_error": "The input must be a valid option, please read the documentation"
}


def convert_errors(e: ValidationError, custom_messages=None) -> list[ErrorDetails]:
    if custom_messages is None:
        custom_messages = CUSTOM_MESSAGES
    new_errors: list[ErrorDetails] = []

    for error in e.errors():
        custom_message = custom_messages.get(error["type"])
        if custom_message:
            ctx = error.get("ctx")
            error["msg"] = custom_message.format(**ctx) if ctx else custom_message
        new_errors.append(error)
    return new_errors


def extract_messages(errors: list[ErrorDetails]) -> list[str]:
    error_messages: list[str] = []
    for error in errors:
        location = error.get("loc")
        parameter_name = location[0]
        parameter_second = location[1] if location[1] else None
        message = f"Configuration Error: Parameter: {parameter_name,parameter_second} - {error['msg']}"
        error_messages.append(message)
    return error_messages


def handle_configuration_errors(e: ValidationError):
    messages = extract_messages(convert_errors(e))
    for message in messages:
        logging.error(message)
