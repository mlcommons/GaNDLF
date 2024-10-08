from huggingface_hub import HfApi, snapshot_download, ModelCardData, ModelCard
from typing import List, Union
from GANDLF import version
from pathlib import Path
from GANDLF.utils import get_git_hash
import re


def validate_model_card(file_path: str):
    """
    Validate that the required fields in the model card are not null, empty, or set to 'REQUIRED_FOR_GANDLF'.
    The fields must contain valid alphabetic or alphanumeric values.

    Args:
        file_path (str): The path to the Markdown file to validate.

    Raises:
        AssertionError: If any required field is missing, empty, null, or contains 'REQUIRED_FOR_GANDLF'.
    """
    # Read the Markdown file
    path = Path(file_path)
    with path.open("r") as file:
        template_str = file.read()

    # Define required fields and their regex patterns to capture the values
    patterns = {
        "Developed by": re.compile(
            r'\*\*Developed by:\*\*\s*\{\{\s*developers\s*\|\s*default\("(.+?)",\s*true\)\s*\}\}',
            re.MULTILINE,
        ),
        "License": re.compile(
            r'\*\*License:\*\*\s*\{\{\s*license\s*\|\s*default\("(.+?)",\s*true\)\s*\}\}',
            re.MULTILINE,
        ),
        "Primary Organization": re.compile(
            r'\*\*Primary Organization:\*\*\s*\{\{\s*primary_organization\s*\|\s*default\("(.+?)",\s*true\)\s*\}\}',
            re.MULTILINE,
        ),
        "Commercial use policy": re.compile(
            r'\*\*Commercial use policy:\*\*\s*\{\{\s*commercial_use\s*\|\s*default\("(.+?)",\s*true\)\s*\}\}',
            re.MULTILINE,
        ),
    }

    # Iterate through the required fields and validate
    for field, pattern in patterns.items():
        match = pattern.search(template_str)

        # Ensure the field is present and does not contain 'REQUIRED_FOR_GANDLF'
        assert match, f"Field '{field}' is missing or not found in the file."

        extract_value = match.group(1)

        # Get the field value
        value = (
            re.search(r"\[([^\]]+)\]", extract_value).group(1)
            if re.search(r"\[([^\]]+)\]", extract_value)
            else None
        )

        # Ensure the field is not set to 'REQUIRED_FOR_GANDLF' or empty
        assert (
            value != "REQUIRED_FOR_GANDLF"
        ), f"The value for '{field}' is set to the default placeholder '[REQUIRED_FOR_GANDLF]'. It must be a valid value."
        assert value, f"The value for '{field}' is empty or null."

        # Ensure the value contains only alphabetic or alphanumeric characters
        assert re.match(
            r"^[a-zA-Z0-9]+$", value
        ), f"The value for '{field}' must be alphabetic or alphanumeric, but got: '{value}'"

    print(
        "All required fields are valid, non-empty, properly filled, and do not contain '[REQUIRED_FOR_GANDLF]'."
    )

    # Example usage
    return template_str


def push_to_model_hub(
    repo_id: str,
    folder_path: str,
    hf_template: str,
    path_in_repo: Union[str, None] = None,
    commit_message: Union[str, None] = None,
    commit_description: Union[str, None] = None,
    token: Union[str, None] = None,
    repo_type: Union[str, None] = None,
    revision: Union[str, None] = None,
    allow_patterns: Union[List[str], str, None] = None,
    ignore_patterns: Union[List[str], str, None] = None,
    delete_patterns: Union[List[str], str, None] = None,
):
    api = HfApi(token=token)

    try:
        repo_id = api.create_repo(repo_id).repo_id
    except Exception as e:
        print(f"Error: {e}")

    tags = ["v" + version]

    git_hash = get_git_hash()

    if not git_hash == "None":
        tags += [git_hash]

    readme_template = validate_model_card(hf_template)

    card_data = ModelCardData(library_name="GaNDLF", tags=tags)
    card = ModelCard.from_template(card_data, template_str=readme_template)

    card.save(Path(folder_path, "README.md"))

    api.upload_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        repo_type="model",
        revision=revision,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        delete_patterns=delete_patterns,
    )
    print("Model Sucessfully Uploded")


def download_from_hub(
    repo_id: str,
    revision: Union[str, None] = None,
    cache_dir: Union[str, None] = None,
    local_dir: Union[str, None] = None,
    force_download: bool = False,
    token: Union[str, None] = None,
):
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        local_dir=local_dir,
        force_download=force_download,
        token=token,
    )
