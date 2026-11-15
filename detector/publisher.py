# detect() 결과를 MQTT로 서버에 보낸다. 사용: publish(detect(1))
import json
import os
import ssl
from dataclasses import asdict

from paho.mqtt import publish as mqtt

from .interface import DetectResult, _load_config


def publish(result: DetectResult) -> None:
    # 전송 실패 시 예외가 그대로 올라간다. 재시도/보관은 호출하는 쪽에서 결정한다.
    cfg = _load_config()["mqtt"]
    username = os.environ.get("MQTT_USERNAME")  # 비밀번호는 저장소에 넣지 않도록 환경변수로 받는다
    # ponytail: 전송마다 새로 연결한다. 발행이 초당 단위로 잦아지면 연결 유지 클라이언트로 교체
    mqtt.single(
        cfg["topic"],
        json.dumps(asdict(result)),
        qos=1,
        hostname=cfg["host"],
        port=cfg["port"],
        auth={"username": username, "password": os.environ.get("MQTT_PASSWORD")} if username else None,
        tls=ssl.create_default_context() if cfg["tls"] else None,
    )
