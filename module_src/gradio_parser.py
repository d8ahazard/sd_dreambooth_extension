import logging
import os.path
import re

logger = logging.getLogger(__name__)

def parse_gr_code(gr_path: str) -> dict:
    outer_stack = {}
    if not os.path.exists(gr_path):
        logger.debug(f"Unable to open file: {gr_path}")
        return outer_stack
    # Define regex patterns for different types of inputs
    checkbox_pattern = re.compile(
        r'(\w+)\s*=\s*gr\.Checkbox\((?:.*?(?:label\s*=\s*"([^"]+)")?.*?)?(?:value\s*=\s*(True|False|None))?\)'
    )
    number_pattern = re.compile(
        r'(\w+) = gr\.Number\((?:.*),\s*value=([\d.e+-]+)\s*(?:,|\))'
    )
    slider_pattern = re.compile(
        r'(\w+) = gr\.Slider\((?:.*),\s*value=([\d.e+-]+)\s*,\s*'
        r'minimum=([\d.e+-]+)\s*,\s*maximum=([\d.e+-]+)\s*,\s*'
        r'step=([\d.e+-]+)\s*(?:,|\))'
    )
    dropdown_pattern = re.compile(
        r'(\w+) = gr\.Dropdown\((?:.*),\s*value=([\w\s]+)\s*,\s*'
        r'choices=(\[.*\])\s*(?:,|\))'
    )
    html_pattern = re.compile(
        r"gr\.HTML\((?:.*),\s*value=\"([\w\s]*)\"\s*\)"
    )
    column_pattern = re.compile(
        r'gr\.(Column|Row)'
    )
    tab_pattern = re.compile(
        r'gr\.(Tab)\((\".*?\")\s*,\s*(\".*?\")'
    )

    parsed_data = {}

    # Initialize stack to keep track of nested columns
    stack = []
    last_tab = ""
    # Process code line by line
    with open(gr_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # Skip comments and empty lines
                if line.strip().startswith('#') or not line.strip():
                    continue

                tab_match = tab_pattern.search(line)
                if tab_match:
                    logger.debug("TAB MATCH")
                    if last_tab != "":
                        outer_stack[last_tab] = parsed_data
                        parsed_data = {}
                        stack = []
                    tab_label = tab_match.group(2)
                    last_tab = tab_label.strip('"')
                    continue

                # Check for column or row start/end
                column_match = column_pattern.search(line)
                if column_match:
                    logger.debug("COL MATCH")
                    column_type = column_match.group(1).lower()
                    if column_type == 'column':
                        stack.append('column')
                        column_num = len(stack)
                        parsed_data[f'column{column_num}'] = {'start': line.strip()}
                    else:
                        column_num = len(stack)
                        parsed_data[f'endColumn{column_num}'] = {'end': line.strip()}
                        stack.pop()
                    continue

                # Check for HTML element
                html_match = html_pattern.search(line)
                if html_match:
                    logger.debug("HTML MATCH")
                    parsed_data[html_match.group(1).strip()] = {
                        'value': html_match.group(1).strip(),
                        'type': 'html',
                        'column': len(stack)
                    }
                    continue

                # Check for checkbox element
                checkbox_match = checkbox_pattern.search(line)
                if checkbox_match:
                    parsed_data[checkbox_match.group(1).strip()] = {
                        'value': checkbox_match.group(2).lower() == 'true',
                        'type': 'checkbox',
                        'column': len(stack)
                    }
                    continue

                # Check for number element
                number_match = number_pattern.search(line)
                if number_match:
                    logger.debug("NUMBER")
                    parsed_data[number_match.group(1).strip()] = {
                        'value': float(number_match.group(2)),
                        'type': 'number',
                        'visible': 'visible' in line,
                        'column': len(stack)
                    }
                    continue

                # Check for slider element
                slider_match = slider_pattern.search(line)
                if slider_match:
                    logger.debug("SLIDER")
                    parsed_data[slider_match.group(1).strip()] = {
                        'value': float(slider_match.group(2)),
                        'type': 'slider',
                        'min': float(slider_match.group(3)),
                        'max': float(slider_match.group(4)),
                        'step': float(slider_match.group(5)),
                        'visible': 'visible' in line,
                        'column': len(stack)
                    }
                    continue

                # Check for dropdown element
                dropdown_match = dropdown_pattern.search(line)
                if dropdown_match:
                    logger.debug("DD")
                    parsed_data[dropdown_match.group(1).strip()] = {
                        'value': dropdown_match.group(2).strip(),
                        'type': 'dropdown',
                        'choices': eval(dropdown_match.group(3).strip()),
                        'column': len(stack)
                    }
                    continue
            except Exception as e:
                logger.debug(f"Error parsing line: {line}")
                logger.debug(e)
    return outer_stack
