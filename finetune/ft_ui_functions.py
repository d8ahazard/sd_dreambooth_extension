import os

from dreambooth import shared
from finetune.configs.workspace_config import WorkspaceConfig
from finetune.convert import extract_checkpoint
from finetune.download import download_checkpoint


def sanitize_folder_name(folder_name):
    """Sanitize a folder name
    Returns:
        sanitized_folder_name
    """
    sanitized_folder_name = folder_name.replace(" ", "_")
    # Remove all non-path characters
    sanitized_folder_name = "".join([c for c in sanitized_folder_name if c.isalnum() or c in ["_", "-", "."]])
    return sanitized_folder_name


def create_ft_workspace(
        ft_new_workspace_name,
        ft_create_from_hub,
        ft_new_model_url,
        ft_new_model_token,
        ft_new_model_src,
        ft_model_type_select
):
    """Create a new workspace for finetuning
    Returns:
        ft_workspace_name,
        ft_model_path,
        ft_has_ema,
        ft_model_type,
        ft_status
    """
    ft_workspace_name = sanitize_folder_name(ft_new_workspace_name)
    # name: str = Field("workspace", title="Workspace name", description="Name of the workspace")
    # base_model: str = Field("", title="Base model", description="Base model to use for training")
    # base_model_type: str = Field("v1", title="Base model type", description="Base model type to use for training")
    # base_model_src: str = Field("local", title="Base model source",
    #                             description="Either 'local' for a user-provided model, or a URL to a HF Model")
    workspace_config = WorkspaceConfig()
    workspace_config.name = ft_workspace_name
    if not ft_create_from_hub:
        workspace_config.base_model = ft_new_model_src
    workspace_config.base_model_src = 'local' if not ft_create_from_hub else ft_new_model_url
    workspace_config.base_model_type = ft_model_type_select
    print(f"workspace_config: {workspace_config}")
    ft_model_path = None
    ft_has_ema = False
    ft_model_type = ft_model_type_select
    workspace_root_path = os.path.join(
        shared.models_path,
        "finetune",
        ft_workspace_name
    )
    if not os.path.exists(workspace_root_path) and not os.path.exists(os.path.join(workspace_root_path, "config.json")):
        print(f"Creating workspace {ft_workspace_name}")
        os.makedirs(workspace_root_path)
        workspace_config.save(ft_workspace_name)
    else:
        print(f"Workspace {ft_workspace_name} already exists")
        ft_status = "Workspace already exists"
        return ft_workspace_name, ft_model_path, ft_has_ema, ft_model_type, ft_status

    if ft_create_from_hub:
        print(f"Downloading checkpoint from hub {ft_new_model_url}")
        output_path = download_checkpoint(ft_new_model_url, ft_new_model_token, ft_model_type)
    else:
        print(f"Extracting checkpoint from {ft_new_model_src}")
        output_path = extract_checkpoint(ft_new_model_src, ft_model_type)
    print(f"output_path: {output_path}")
    return ft_workspace_name, ft_model_path, ft_has_ema, ft_model_type, "Success"
