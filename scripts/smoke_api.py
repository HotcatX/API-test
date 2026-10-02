"""Check a running API, including real vector inference, using only stdlib."""

import argparse
import json
import math
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request(base_url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=30) as response:
            return response.status, response.read()
    except HTTPError as exc:
        return exc.code, exc.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    for attempt in range(60):
        try:
            status, body = request(args.base_url, "/")
            if status == 200:
                break
        except (URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(1)
    else:
        raise SystemExit("The API did not become ready within 60 attempts.")

    assert json.loads(body) == {"Hello": "World"}
    print("PASS GET / (200)")
    status, body = request(args.base_url, "/embedding?word=apple")
    assert status == 200, (status, body)
    result = json.loads(body)
    vector = result["embedding"]
    assert result["word"] == "apple"
    assert result["model"] == "en_core_web_md"
    assert result["dimensions"] == len(vector) == 300
    assert all(isinstance(value, (float, int)) and math.isfinite(value) for value in vector)
    assert any(value != 0 for value in vector)
    print("PASS GET /embedding?word=apple (300 finite, nonzero vector values)")
    status, body = request(args.base_url, "/embedding?word=qzxwvqzxwvqzxwv")
    assert status == 422, (status, body)
    print("PASS unknown word rejected (422)")
    status, body = request(args.base_url, "/generate", {"start_word": "we", "length": 3})
    assert status == 200, (status, body)
    assert json.loads(body)["generated_text"] in {"we are generating", "we are simple"}
    print("PASS POST /generate (200)")
    status, body = request(args.base_url, "/docs")
    assert status == 200 and b"SwaggerUIBundle" in body
    print("PASS GET /docs (200)")


if __name__ == "__main__":
    main()
