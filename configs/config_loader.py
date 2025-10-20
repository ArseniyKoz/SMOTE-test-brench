from pathlib import Path
import yaml


class ConfigLoader:
    def __init__(self, config_name: str = "config.yaml"):

        self.script_dir = Path(__file__).resolve().parent
        self.config_path = self.script_dir / config_name

    def load(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with self.config_path.open('r', encoding='utf-8') as f:
            return yaml.safe_load(f)


