XRAY_CONFIG = {
    "version": "3.4b3\n",
    "mode": "blocks",
    "dev_mode": True,
    "analytics_enabled": False,
    "components": [
        {
            "id": 27,
            "type": "markdown",
            "props": {
                "value": "<h1>Detect Disease From Scan</h1>\n<p>With this model you can lorem ipsum</p>\n<ul>\n<li>ipsum 1</li>\n<li>ipsum 2</li>\n</ul>\n",
                "name": "markdown",
                "visible": True,
                "style": {},
            },
        },
        {
            "id": 28,
            "type": "checkboxgroup",
            "props": {
                "choices": ["Covid", "Malaria", "Lung Cancer"],
                "value": [],
                "label": "Disease to Scan For",
                "show_label": True,
                "name": "checkboxgroup",
                "visible": True,
                "style": {},
            },
        },
        {"id": 29, "type": "tabs", "props": {"visible": True, "style": {}}},
        {
            "id": 30,
            "type": "tabitem",
            "props": {"label": "X-ray", "visible": True, "style": {}},
        },
        {
            "id": 31,
            "type": "row",
            "props": {
                "type": "row",
                "variant": "default",
                "visible": True,
                "style": {},
            },
        },
        {
            "id": 32,
            "type": "image",
            "props": {
                "image_mode": "RGB",
                "source": "upload",
                "tool": "editor",
                "streaming": False,
                "mirror_webcam": True,
                "show_label": True,
                "name": "image",
                "visible": True,
                "style": {},
            },
        },
        {
            "id": 33,
            "type": "json",
            "props": {"show_label": True, "name": "json", "visible": True, "style": {}},
        },
        {
            "id": 34,
            "type": "button",
            "props": {
                "value": "Run",
                "variant": "secondary",
                "interactive": True,
                "name": "button",
                "visible": True,
                "style": {},
            },
        },
        {
            "id": 35,
            "type": "tabitem",
            "props": {"label": "CT Scan", "visible": True, "style": {}},
        },
        {
            "id": 36,
            "type": "row",
            "props": {
                "type": "row",
                "variant": "default",
                "visible": True,
                "style": {},
            },
        },
        {
            "id": 37,
            "type": "image",
            "props": {
                "image_mode": "RGB",
                "source": "upload",
                "tool": "editor",
                "streaming": False,
                "mirror_webcam": True,
                "show_label": True,
                "name": "image",
                "visible": True,
                "style": {},
            },
        },
        {
            "id": 38,
            "type": "json",
            "props": {"show_label": True, "name": "json", "visible": True, "style": {}},
        },
        {
            "id": 39,
            "type": "button",
            "props": {
                "value": "Run",
                "variant": "secondary",
                "name": "button",
                "interactive": True,
                "visible": True,
                "style": {},
            },
        },
        {
            "id": 40,
            "type": "textbox",
            "props": {
                "lines": 1,
                "max_lines": 20,
                "value": "",
                "type": "text",
                "show_label": True,
                "name": "textbox",
                "visible": True,
                "style": {},
            },
        },
        {
            "id": 41,
            "type": "form",
            "props": {"type": "form", "visible": True, "style": {}},
        },
        {
            "id": 42,
            "type": "form",
            "props": {"type": "form", "visible": True, "style": {}},
        },
    ],
    "css": None,
    "title": "Gradio",
    "is_space": False,
    "enable_queue": None,
    "show_error": False,
    "show_api": True,
    "layout": {
        "id": 26,
        "children": [
            {"id": 27},
            {"id": 41, "children": [{"id": 28}]},
            {
                "id": 29,
                "children": [
                    {
                        "id": 30,
                        "children": [
                            {"id": 31, "children": [{"id": 32}, {"id": 33}]},
                            {"id": 34},
                        ],
                    },
                    {
                        "id": 35,
                        "children": [
                            {"id": 36, "children": [{"id": 37}, {"id": 38}]},
                            {"id": 39},
                        ],
                    },
                ],
            },
            {"id": 42, "children": [{"id": 40}]},
        ],
    },
    "dependencies": [
        {
            "targets": [34],
            "trigger": "click",
            "inputs": [28, 32],
            "outputs": [33],
            "backend_fn": True,
            "js": None,
            "queue": None,
            "api_name": None,
            "scroll_to_output": False,
            "show_progress": True,
            "batch": False,
            "max_batch_size": 4,
            "cancels": [],
            "every": None,
            "collects_event_data": False,
            "types": {"continuous": False, "generator": False},
            "trigger_after": None,
            "trigger_only_on_success": False,
        },
        {
            "targets": [39],
            "trigger": "click",
            "inputs": [28, 37],
            "outputs": [38],
            "backend_fn": True,
            "js": None,
            "queue": None,
            "api_name": None,
            "scroll_to_output": False,
            "show_progress": True,
            "batch": False,
            "max_batch_size": 4,
            "cancels": [],
            "every": None,
            "collects_event_data": False,
            "types": {"continuous": False, "generator": False},
            "trigger_after": None,
            "trigger_only_on_success": False,
        },
        {
            "targets": [],
            "trigger": "load",
            "inputs": [],
            "outputs": [40],
            "backend_fn": True,
            "js": None,
            "queue": None,
            "api_name": None,
            "scroll_to_output": False,
            "show_progress": True,
            "batch": False,
            "max_batch_size": 4,
            "cancels": [],
            "every": None,
            "collects_event_data": False,
            "types": {"continuous": False, "generator": False},
            "trigger_after": None,
            "trigger_only_on_success": False,
        },
    ],
}


XRAY_CONFIG_DIFF_IDS = {
    "version": "3.4b3\n",
    "mode": "blocks",
    "analytics_enabled": False,
    "dev_mode": True,
    "components": [
        {
            "id": 27,
            "type": "markdown",
            "props": {
                "value": "<h1>Detect Disease From Scan</h1>\n<p>With this model you can lorem ipsum</p>\n<ul>\n<li>ipsum 1</li>\n<li>ipsum 2</li>\n</ul>\n",
                "name": "markdown",
                "visible": True,
                "style": {},
            },
        },
        {
            "id": 28,
            "type": "checkboxgroup",
            "props": {
                "choices": ["Covid", "Malaria", "Lung Cancer"],
                "value": [],
                "label": "Disease to Scan For",
                "show_label": True,
                "name": "checkboxgroup",
                "visible": True,
                "style": {},
            },
        },
        {"id": 29, "type": "tabs", "props": {"visible": True, "style": {}}},
        {
            "id": 30,
            "type": "tabitem",
            "props": {"label": "X-ray", "visible": True, "style": {}},
        },
        {
            "id": 31,
            "type": "row",
            "props": {
                "type": "row",
                "variant": "default",
                "visible": True,
                "style": {},
            },
        },
        {
            "id": 32,
            "type": "image",
            "props": {
                "image_mode": "RGB",
                "source": "upload",
                "tool": "editor",
                "streaming": False,
                "mirror_webcam": True,
                "show_label": True,
                "name": "image",
                "visible": True,
                "style": {},
            },
        },
        {
            "id": 33,
            "type": "json",
            "props": {"show_label": True, "name": "json", "visible": True, "style": {}},
        },
        {
            "id": 34,
            "type": "button",
            "props": {
                "value": "Run",
                "variant": "secondary",
                "interactive": True,
                "name": "button",
                "visible": True,
                "style": {},
            },
        },
        {
            "id": 35,
            "type": "tabitem",
            "props": {"label": "CT Scan", "visible": True, "style": {}},
        },
        {
            "id": 36,
            "type": "row",
            "props": {
                "type": "row",
                "variant": "default",
                "visible": True,
                "style": {},
            },
        },
        {
            "id": 37,
            "type": "image",
            "props": {
                "image_mode": "RGB",
                "source": "upload",
                "tool": "editor",
                "streaming": False,
                "mirror_webcam": True,
                "show_label": True,
                "name": "image",
                "visible": True,
                "style": {},
            },
        },
        {
            "id": 38,
            "type": "json",
            "props": {"show_label": True, "name": "json", "visible": True, "style": {}},
        },
        {
            "id": 933,
            "type": "button",
            "props": {
                "value": "Run",
                "variant": "secondary",
                "interactive": True,
                "name": "button",
                "visible": True,
                "style": {},
            },
        },
        {
            "id": 40,
            "type": "textbox",
            "props": {
                "lines": 1,
                "max_lines": 20,
                "value": "",
                "type": "text",
                "show_label": True,
                "name": "textbox",
                "visible": True,
                "style": {},
            },
        },
        {
            "id": 41,
            "type": "form",
            "props": {"type": "form", "visible": True, "style": {}},
        },
        {
            "id": 42,
            "type": "form",
            "props": {"type": "form", "visible": True, "style": {}},
        },
    ],
    "css": None,
    "title": "Gradio",
    "is_space": False,
    "enable_queue": None,
    "show_error": False,
    "show_api": True,
    "layout": {
        "id": 26,
        "children": [
            {"id": 27},
            {"id": 41, "children": [{"id": 28}]},
            {
                "id": 29,
                "children": [
                    {
                        "id": 30,
                        "children": [
                            {"id": 31, "children": [{"id": 32}, {"id": 33}]},
                            {"id": 34},
                        ],
                    },
                    {
                        "id": 35,
                        "children": [
                            {"id": 36, "children": [{"id": 37}, {"id": 38}]},
                            {"id": 933},
                        ],
                    },
                ],
            },
            {"id": 42, "children": [{"id": 40}]},
        ],
    },
    "dependencies": [
        {
            "targets": [34],
            "trigger": "click",
            "inputs": [28, 32],
            "outputs": [33],
            "backend_fn": True,
            "js": None,
            "queue": None,
            "api_name": None,
            "scroll_to_output": False,
            "show_progress": True,
            "batch": False,
            "max_batch_size": 4,
            "cancels": [],
            "every": None,
            "collects_event_data": False,
            "types": {"continuous": False, "generator": False},
            "trigger_after": None,
            "trigger_only_on_success": False,
        },
        {
            "targets": [933],
            "trigger": "click",
            "inputs": [28, 37],
            "outputs": [38],
            "backend_fn": True,
            "js": None,
            "queue": None,
            "api_name": None,
            "scroll_to_output": False,
            "show_progress": True,
            "batch": False,
            "max_batch_size": 4,
            "cancels": [],
            "every": None,
            "collects_event_data": False,
            "types": {"continuous": False, "generator": False},
            "trigger_after": None,
            "trigger_only_on_success": False,
        },
        {
            "targets": [],
            "trigger": "load",
            "inputs": [],
            "outputs": [40],
            "backend_fn": True,
            "js": None,
            "queue": None,
            "api_name": None,
            "scroll_to_output": False,
            "show_progress": True,
            "batch": False,
            "max_batch_size": 4,
            "cancels": [],
            "every": None,
            "collects_event_data": False,
            "types": {"continuous": False, "generator": False},
            "trigger_after": None,
            "trigger_only_on_success": False,
        },
    ],
}


XRAY_CONFIG_WITH_MISTAKE = {
    "mode": "blocks",
    "dev_mode": True,
    "analytics_enabled": False,
    "components": [
        {
            "id": 1,
            "type": "markdown",
            "props": {
                "value": "<h1>Detect Disease From Scan</h1>\n<p>With this model you can lorem ipsum</p>\n<ul>\n<li>ipsum 1</li>\n<li>ipsum 2</li>\n</ul>\n",
                "name": "markdown",
                "style": {},
            },
        },
        {
            "id": 2,
            "type": "checkboxgroup",
            "props": {
                "choices": ["Covid", "Malaria", "Lung Cancer"],
                "value": [],
                "name": "checkboxgroup",
                "show_label": True,
                "label": "Disease to Scan For",
                "style": {},
            },
        },
        {
            "id": 3,
            "type": "tabs",
            "props": {
                "style": {},
                "value": True,
            },
        },
        {
            "id": 4,
            "type": "tabitem",
            "props": {
                "label": "X-ray",
                "style": {},
                "value": True,
            },
        },
        {
            "id": 5,
            "type": "row",
            "props": {"type": "row", "variant": "default", "style": {}, "value": True},
        },
        {
            "id": 6,
            "type": "image",
            "props": {
                "image_mode": "RGB",
                "source": "upload",
                "streaming": False,
                "mirror_webcam": True,
                "tool": "editor",
                "name": "image",
                "style": {},
            },
        },
        {
            "id": 7,
            "type": "json",
            "props": {
                "name": "json",
                "style": {},
            },
        },
        {
            "id": 8,
            "type": "button",
            "props": {
                "value": "Run",
                "name": "button",
                "interactive": True,
                "css": {"background-color": "red", "--hover-color": "orange"},
                "variant": "secondary",
            },
        },
        {
            "id": 9,
            "type": "tabitem",
            "props": {
                "show_label": True,
                "label": "CT Scan",
                "style": {},
                "value": True,
            },
        },
        {
            "id": 10,
            "type": "row",
            "props": {"type": "row", "variant": "default", "style": {}, "value": True},
        },
        {
            "id": 11,
            "type": "image",
            "props": {
                "image_mode": "RGB",
                "source": "upload",
                "tool": "editor",
                "streaming": False,
                "mirror_webcam": True,
                "name": "image",
                "style": {},
            },
        },
        {
            "id": 12,
            "type": "json",
            "props": {
                "name": "json",
                "style": {},
            },
        },
        {
            "id": 13,
            "type": "button",
            "props": {
                "value": "Run",
                "interactive": True,
                "name": "button",
                "style": {},
                "variant": "secondary",
            },
        },
        {
            "id": 14,
            "type": "textbox",
            "props": {
                "lines": 1,
                "value": "",
                "name": "textbox",
                "type": "text",
                "style": {},
            },
        },
    ],
    "layout": {
        "id": 0,
        "children": [
            {"id": 1},
            {"id": 2},
            {
                "id": 3,
                "children": [
                    {
                        "id": 4,
                        "children": [
                            {"id": 5, "children": [{"id": 6}, {"id": 7}]},
                            {"id": 8},
                        ],
                    },
                    {
                        "id": 9,
                        "children": [
                            {"id": 10, "children": [{"id": 12}, {"id": 11}]},
                            {"id": 13},
                        ],
                    },
                ],
            },
            {"id": 14},
        ],
    },
    "dependencies": [
        {
            "targets": [8],
            "trigger": "click",
            "inputs": [2, 6],
            "outputs": [7],
            "api_name": None,
            "scroll_to_output": False,
            "show_progress": True,
            "cancels": [],
            "trigger_after": None,
            "trigger_only_on_success": False,
        },
        {
            "targets": [13],
            "trigger": "click",
            "inputs": [2, 11],
            "outputs": [12],
            "api_name": None,
            "scroll_to_output": False,
            "show_progress": True,
            "cancels": [],
            "trigger_after": None,
            "trigger_only_on_success": False,
        },
    ],
}