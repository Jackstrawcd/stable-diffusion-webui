import os

import network
import networks

from modules import shared, ui_extra_networks
from modules.ui_extra_networks import quote_js
from ui_edit_user_metadata import LoraUserMetadataEditor


user_lora_dir = os.path.join(os.path.dirname(os.path.dirname(shared.cmd_opts.lora_dir)), "user-models", 'Lora')
if not os.path.isdir(user_lora_dir):
    user_lora_dir = None

class ExtraNetworksPageLora(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Lora')

    def refresh(self):
        networks.list_available_networks()

    def create_item(self, name, index=None, enable_filter=True):
        lora_on_disk = networks.available_networks.get(name)

        path, ext = os.path.splitext(lora_on_disk.filename)

        alias = lora_on_disk.get_alias()

        item = {
            "name": name,
            "filename": lora_on_disk.filename,
            "shorthash": lora_on_disk.shorthash,
            "preview": self.find_preview(path),
            "description": self.find_description(path),
            "search_term": self.search_terms_from_path(lora_on_disk.filename) + " " + (lora_on_disk.hash or ""),
            "local_preview": f"{path}.{shared.opts.samples_format}",
            "metadata": lora_on_disk.metadata,
            "sort_keys": {'default': index, **self.get_sort_keys(lora_on_disk.filename)},
            "sd_version": lora_on_disk.sd_version.name,
        }

        self.read_user_metadata(item)
        activation_text = item["user_metadata"].get("activation text")
        preferred_weight = item["user_metadata"].get("preferred weight", 0.0)
        item["prompt"] = quote_js(f"<lora:{alias}:") + " + " + (str(preferred_weight) if preferred_weight else "opts.extra_networks_default_multiplier") + " + " + quote_js(">")

        if activation_text:
            item["prompt"] += " + " + quote_js(" " + activation_text)

        sd_version = item["user_metadata"].get("sd version")
        if sd_version in network.SdVersion.__members__:
            item["sd_version"] = sd_version
            sd_version = network.SdVersion[sd_version]
        else:
            sd_version = lora_on_disk.sd_version

        if shared.opts.lora_show_all or not enable_filter:
            pass
        elif sd_version == network.SdVersion.Unknown:
            model_version = network.SdVersion.SDXL if shared.sd_model.is_sdxl else network.SdVersion.SD2 if shared.sd_model.is_sd2 else network.SdVersion.SD1
            if model_version.name in shared.opts.lora_hide_unknown_for_versions:
                return None
        elif shared.sd_model.is_sdxl and sd_version != network.SdVersion.SDXL:
            return None
        elif shared.sd_model.is_sd2 and sd_version != network.SdVersion.SD2:
            return None
        elif shared.sd_model.is_sd1 and sd_version != network.SdVersion.SD1:
            return None

        return item

    def list_items(self):
        for index, name in enumerate(networks.available_networks):
            item = self.create_item(name, index)

            if item is not None:
                yield item

    def allowed_directories_for_previews(self):
        user_lora_dir = os.path.join(os.path.dirname(os.path.dirname(shared.cmd_opts.lora_dir)), "user-models", 'Lora')
        user_lycoris_dir = os.path.join(os.path.dirname(os.path.dirname(shared.cmd_opts.lora_dir)), "user-models",
                                        'Lycoris')
        return [shared.cmd_opts.lora_dir,
                shared.cmd_opts.lyco_dir_backcompat,
                user_lora_dir,
                user_lycoris_dir]

    def create_user_metadata_editor(self, ui, tabname):
        return LoraUserMetadataEditor(ui, tabname, self)
