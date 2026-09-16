"""Build both images, replace the containers, then verify the deployed model."""

import json
import logging
from pathlib import Path
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from penguins import EXAMPLE, ROOT, write_json

LOGGER = logging.getLogger("deploy")


def deploy() -> None:
    metadata = json.loads((ROOT / "models/metadata.json").read_text(encoding="utf-8"))
    subprocess.run(["docker", "compose", "-f", str(ROOT / "code/deployment/docker-compose.yml"),
                    "up", "--build", "--force-recreate", "--wait", "--wait-timeout", "120"],
                   cwd=ROOT, check=True)
    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=10) as response:
        health = json.load(response)
    request = urllib.request.Request("http://127.0.0.1:8000/predict", data=json.dumps(EXAMPLE).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        prediction = json.load(response)
    if health["run_id"] != metadata["run_id"] or prediction["run_id"] != metadata["run_id"]:
        raise RuntimeError("API is not serving the model produced by this training run.")
    write_json(ROOT / ".runtime/deployment.json", {"health": health, "example_prediction": prediction})
    LOGGER.info("Deployment verified: run_id=%s species=%s", health["run_id"], prediction["species"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    deploy()
