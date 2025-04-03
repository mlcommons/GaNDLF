import logging
from typing import Optional, Union


from typing import Type
from pydantic import BaseModel, ValidationError, create_model
from pydantic_core import ErrorDetails


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


# Define custom error messages. The key must be a pydantic type error.
CUSTOM_MESSAGES = {
    "literal_error": "The input must be a valid option, please read the documentation",
    "missing": "This parameter is required. Please define it",
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
        if len(location) == 2:
            message = f"Configuration Error: Parameter: ({location[0]}, {location[1]}) - {error['msg']}"
        else:
            message = (
                f"Configuration Error: Parameter: ({location[0]}) - {error['msg']}"
            )
        error_messages.append(message)
    return error_messages


def handle_configuration_errors(e: ValidationError):
    messages = extract_messages(convert_errors(e))
    for message in messages:
        logging.error(message)


def combine_models(base_model: Type[BaseModel], extra_model: Type[BaseModel]):
    """Combine base model with an extra model dynamically."""
    fields = {}
    # Collect base model fields
    for field_name, field_info in base_model.model_fields.items():
        fields[field_name] = (
            field_info.annotation,
            field_info.default if field_info.default is not Ellipsis else ...,
        )

    # Add fields from the extra model
    for field_name, field_info in extra_model.model_fields.items():
        fields[field_name] = (
            field_info.annotation,
            field_info.default if field_info.default is not Ellipsis else ...,
        )

    # Return the new dynamically combined model
    return create_model(base_model.__name__, **fields)
