import sys
import platform
import logging
import torch
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

DEFAULT_NAME = "NeuroPilot"
console = Console()

# Configure module-level singleton logger
logger = logging.getLogger(DEFAULT_NAME)
logger.setLevel(logging.INFO)
logger.propagate = True

if not logger.handlers:
    console_handler = RichHandler(
        show_time=False,
        show_path=False,
        markup=True,
        rich_tracebacks=True
    )
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

def set_logger(name=DEFAULT_NAME, save_dir=None, endpoint=None):
    """
    Configure logger with custom format mimicking Ultralytics.
    """
    if name != DEFAULT_NAME:
        logger.name = name

    # Remove old file handlers if any
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        log_file = save_dir / "neuropilot.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    return logger


def log_system_info():
    """Logs system information similar to Ultralytics startup."""
    try:
        os_info = f"{platform.system()} {platform.release()}"
        python_info = f"Python-{platform.python_version()}"

        torch_info = f"torch-{torch.__version__}"
        if torch.cuda.is_available():
            devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            cuda_info = f"CUDA:{torch.version.cuda} ({', '.join(devices)})"
        else:
            cuda_info = "CPU"

        logger.info(f"System: {os_info}")
        logger.info(f"Environment: {python_info} {torch_info} {cuda_info}")

    except Exception as e:
        logger.warning(f"Failed to log system info: {e}")

def log_banner(title, subtitle=""):
    """Hiển thị banner xịn bằng rich, đồng thời log dưới dạng text."""
    console.print(Panel.fit(f"[bold cyan]{title}[/]\n[dim]{subtitle}[/]", border_style="cyan"))
    logger.info(f"=== {title} | {subtitle} ===")

def log_table(title, data_dict):
    """Hiển thị table xịn bằng rich, đồng thời log dưới dạng text."""
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    for k, v in data_dict.items():
        val_str = f"{v:.4f}" if isinstance(v, (float, int)) else str(v)
        table.add_row(str(k), val_str)
        logger.info(f"{title} - {k}: {val_str}")
    console.print(table)


def colorstr(*args):
    """Color a string using ANSI escape codes (e.g., colorstr('blue', 'bold', 'hello'))."""
    *styles, string = args if len(args) > 1 else ("blue", "bold", args[0])
    colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors.get(x, "") for x in styles) + f"{string}" + colors["end"]


__all__ = [
    "logger",
    "set_logger",
    "log_system_info",
    "console",
    "log_banner",
    "log_table",
    "colorstr",
]
