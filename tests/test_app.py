from unittest.mock import Mock, patch

import requests
from streamlit.testing.v1 import AppTest

from conftest import ROOT


def test_app_displays_api_prediction():
    response = Mock()
    response.json.return_value = {"species": "Adelie", "run_id": "test-run"}
    with patch("requests.post", return_value=response) as post:
        app = AppTest.from_file(str(ROOT / "code/deployment/app/app.py")).run()
        assert len(app.number_input) == 4
        app.button[0].click().run()
    assert not app.exception
    assert app.success[0].value == "Predicted species: Adelie"
    assert post.call_args.kwargs["timeout"] == 10
    assert "test-run" in app.caption[0].value


def test_app_explains_api_unavailability():
    with patch("requests.post", side_effect=requests.ConnectionError("offline")):
        app = AppTest.from_file(str(ROOT / "code/deployment/app/app.py")).run()
        app.button[0].click().run()
    assert not app.exception
    assert "temporarily unavailable" in app.error[0].value
