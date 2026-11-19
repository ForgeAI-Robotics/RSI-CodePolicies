"""Persistent SSH transport to the existing native Isaac file-RPC worker.

Only typed measurements cross into the policy. Privileged native diagnostics stay
in host audit files and the environment success predicate.
"""

from __future__ import annotations

import json
import os
import select
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .learning import file_hash, write_new_json
from .scenarios import ROOT, load_protocol
from .types import Action, Observation, PersonTrack, SensorProvenance, SurfacePatch


@dataclass(frozen=True)
class RemoteConfig:
    host: str = "ubuntu@10.60.71.213"
    key: str = "/data1/liuming/forgeai.pem"
    root: str = "/mnt/shared/moonten/RSI"
    user: str = "moonten"
    gpu: int = 0
    sync_code: bool = False
    startup_timeout_s: float = 600.0
    rpc_timeout_s: float = 300.0

    @classmethod
    def from_environment(cls) -> RemoteConfig:
        return cls(
            host=os.environ.get("CHECKPOINT_REMOTE_HOST", cls.host),
            key=os.environ.get("CHECKPOINT_REMOTE_KEY", cls.key),
            root=os.environ.get("CHECKPOINT_REMOTE_ROOT", cls.root),
            user=os.environ.get("CHECKPOINT_REMOTE_USER", cls.user),
            gpu=int(os.environ.get("CHECKPOINT_GPU", "0")),
            sync_code=os.environ.get("CHECKPOINT_SYNC_CODE") == "1",
        )


def decode_observation(payload: dict[str, Any]) -> Observation:
    """Decode the dataclass wire contract without admitting privileged extras."""

    def patch(value):
        if value is None:
            return None
        return SurfacePatch(
            center=tuple(value["center"]),
            normal=tuple(value["normal"]),
            tangent=tuple(value["tangent"]),
            confidence=value["confidence"],
            stamp=value["stamp"],
        )

    data = dict(payload)
    data["people"] = tuple(
        PersonTrack(
            track_id=p["track_id"],
            position=tuple(p["position"]),
            forward=tuple(p["forward"]),
            velocity=tuple(p["velocity"]),
            stamp=p["stamp"],
            bag=patch(p.get("bag")),
            forearm=patch(p.get("forearm")),
            in_queue=p.get("in_queue", False),
            in_waiting_area=p.get("in_waiting_area", False),
        )
        for p in data.get("people", [])
    )
    data["sensors"] = {
        name: SensorProvenance(**value) for name, value in data.get("sensors", {}).items()
    }
    for name in ("paper_grasp", "supply_exit", "deposit"):
        data[name] = patch(data.get(name))
    for name in ("base_pose", "eef_position", "eef_rpy", "paper_tip"):
        if data.get(name) is not None:
            data[name] = tuple(data[name])
    return Observation(**data)


class RemoteNativeBackend:
    def __init__(self, out_dir: Path, binding: dict, config: RemoteConfig):
        self.output, self.binding, self.config = out_dir, binding, config
        self.remote = f"{config.root}/runs/airport_checkpoint_native/{binding['run_id']}"
        self.launch = f"{config.root}/runs/airport_checkpoint_launch/{binding['run_id']}"
        self._bridge = None
        self._bridge_errors = None
        self._closed = False
        self._lock = threading.Lock()

    def _ssh(self, argv: list[str]) -> list[str]:
        return [
            "ssh",
            "-i",
            self.config.key,
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            self.config.host,
            shlex.join(argv),
        ]

    def _python(self, source: str, timeout: float = 60, *, administrator: bool = False) -> dict:
        identity = ["sudo", "-n"]
        if not administrator:
            identity.extend(["-u", self.config.user])
        result = subprocess.check_output(
            self._ssh([*identity, "python3", "-c", source]),
            text=True,
            timeout=timeout,
        )
        return json.loads(result)

    def _rsync(self, source: str, destination: str) -> None:
        subprocess.run(
            [
                "rsync",
                "-a",
                "--exclude=__pycache__",
                "--exclude=*.pyc",
                "--rsync-path=" + shlex.join(["sudo", "-n", "-u", self.config.user, "rsync"]),
                "-e",
                shlex.join(["ssh", "-i", self.config.key, "-o", "BatchMode=yes"]),
                source,
                destination,
            ],
            check=True,
            timeout=300,
        )

    def prepare_and_start(self) -> None:
        config = self.config
        container = f"rsi-airport-isaac-{config.gpu}"
        # The launcher also has an atomic Docker name guard. Never stop another
        # experiment just to acquire the GPU.
        self._python(
            f"""import json,subprocess
r=subprocess.run(['sudo','-n','docker','inspect','--format','{{{{.State.Running}}}}',{container!r}],capture_output=True,text=True)
if r.returncode==0 and r.stdout.strip()=='true':
 raise RuntimeError('Checkpoint GPU already has an active worker')
print(json.dumps({{'gpu_available':True}}))
""",
            administrator=True,
        )
        self._python(f"""import json
from pathlib import Path
Path({self.launch!r}).mkdir(parents=True,exist_ok=False)
if Path({self.remote!r}).exists():raise FileExistsError('Remote episode already exists')
skill=Path({(config.root + "/airport/workflows/airport-checkpoint-workflow")!r})
skill.mkdir(parents=True,exist_ok=True)
print(json.dumps({{'prepared':True}}))
""")
        if config.sync_code:
            self._rsync(
                str(ROOT / "airport/checkpoint") + "/",
                f"{config.host}:{config.root}/airport/checkpoint/",
            )
            for relative in (
                "scripts/checkpoint_native_worker.py",
                "scripts/run_airport_isaac.sh",
                "airport/workflows/airport-checkpoint-workflow/SKILL.md",
            ):
                self._rsync(str(ROOT / relative), f"{config.host}:{config.root}/{relative}")
        for name in ("scenario.json", "protocol.json", "checkpoint_binding.json", "candidate.json"):
            self._rsync(str(self.output / name), f"{config.host}:{self.launch}/{name}")
        candidate = json.loads((self.output / "candidate.json").read_text())
        self._python(f"""import hashlib,json
from pathlib import Path
root=Path({config.root!r})
files={candidate["files"]!r}
for name,expected in files.items():
 p=root/'airport/checkpoint'/name
 if name=='SKILL.md':p=root/'airport/workflows/airport-checkpoint-workflow/SKILL.md'
 if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
  raise RuntimeError('Remote source mismatch: '+name)
print(json.dumps({{'source_verified':True}}))
""")
        protocol = load_protocol()
        argv = [
            "bash",
            f"{config.root}/scripts/run_airport_isaac.sh",
            "--mode",
            "rpc",
            "--out",
            self.remote,
            "--scenario",
            f"{self.launch}/scenario.json",
            "--protocol",
            f"{self.launch}/protocol.json",
            "--binding",
            f"{self.launch}/checkpoint_binding.json",
            "--seed",
            str(self.binding["seed"]),
            "--scenario-id",
            self.binding["scenario_id"],
            "--rpc-timeout-s",
            str(protocol["wall_budget_s"] + 120),
        ]
        self._python(
            f"""import json,os,subprocess
from pathlib import Path
env=dict(os.environ,ISAAC_GPU_INDEX={str(config.gpu)!r},ISAAC_ENTRY_SCRIPT='/work/scripts/checkpoint_native_worker.py')
with open({(self.launch + "/worker.log")!r},'wb') as log:
 p=subprocess.Popen({argv!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
Path({(self.launch + "/worker.pid")!r}).write_text(str(p.pid))
os.chown({(self.launch + "/worker.log")!r},1004,1004)
os.chown({(self.launch + "/worker.pid")!r},1004,1004)
print(json.dumps({{'pid':p.pid}}))
""",
            administrator=True,
        )
        deadline = time.monotonic() + config.startup_timeout_s
        while time.monotonic() < deadline:
            status = self._python(f"""import json
from pathlib import Path
p=Path({self.remote!r})
error=p/'error.txt'
ready=p/'ready.json'
pid=Path({(self.launch + "/worker.pid")!r}).read_text().strip()
proc=Path('/proc')/pid/'stat'
alive=proc.exists() and proc.read_text().split(') ',1)[1][0]!='Z'
log=Path({(self.launch + "/worker.log")!r})
result={{'error':error.read_text() if error.exists() else None,
        'ready':ready.exists() and (p/'rpc').is_dir(), 'alive':alive,
        'log_tail':log.read_text(errors='replace')[-4000:] if not alive else ''}}
print(json.dumps(result))
""")
            if status["error"]:
                self.sync_artifacts()
                raise RuntimeError(status["error"])
            if status["ready"]:
                self._start_bridge()
                return
            if not status["alive"]:
                self.sync_artifacts()
                raise RuntimeError("Native worker exited before readiness: " + status["log_tail"])
            time.sleep(2)
        self._stop_own_worker()
        self.sync_artifacts()
        raise TimeoutError("Native worker did not become ready; inspect remote_worker.log")

    def _stop_own_worker(self) -> None:
        """Stop only the container whose --out argument is this fresh episode."""
        container = f"rsi-airport-isaac-{self.config.gpu}"
        self._python(
            f"""import json,subprocess
r=subprocess.run(['sudo','-n','docker','inspect',{container!r}],capture_output=True,text=True)
stopped=False
if r.returncode==0:
 data=json.loads(r.stdout)[0]
 if {self.remote!r} in data.get('Args',[]):
  subprocess.run(['sudo','-n','docker','stop','--time','10',{container!r}],check=True,capture_output=True)
  stopped=True
print(json.dumps({{'stopped_own_worker':stopped}}))
""",
            timeout=30,
            administrator=True,
        )

    def _start_bridge(self) -> None:
        source = f"""import json,sys,time,uuid
from pathlib import Path
queue=Path({(self.remote + "/rpc")!r})
for line in sys.stdin:
 request=json.loads(line)
 ident=uuid.uuid4().hex
 target=queue/(ident+'.request.json')
 temporary=target.with_suffix('.part')
 temporary.write_text(json.dumps(request))
 temporary.replace(target)
 response=queue/(ident+'.response.json')
 deadline=time.monotonic()+{self.config.rpc_timeout_s!r}
 while not response.exists():
  if time.monotonic()>deadline:break
  time.sleep(.005)
 result={{'ok':False,'error':'Native RPC timeout'}}
 if response.exists():result=json.loads(response.read_text())
 print(json.dumps(result),flush=True)
"""
        self._bridge_errors = (self.output / "remote_bridge.stderr.log").open("wb")
        self._bridge = subprocess.Popen(
            self._ssh(["sudo", "-n", "-u", self.config.user, "python3", "-u", "-c", source]),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._bridge_errors,
            text=True,
            bufsize=1,
        )

    def _rpc(self, name: str, arguments: dict | None = None) -> Any:
        with self._lock:
            if self._bridge is None or self._bridge.poll() is not None:
                raise RuntimeError("Native RPC bridge is unavailable")
            started = time.monotonic()
            self._bridge.stdin.write(
                json.dumps({"name": name, "arguments": arguments or {}}) + "\n"
            )
            self._bridge.stdin.flush()
            ready, _, _ = select.select(
                [self._bridge.stdout], [], [], self.config.rpc_timeout_s + 10
            )
            if not ready:
                self._bridge.terminate()
                raise TimeoutError(f"Native RPC {name} exceeded deadline")
            response = json.loads(self._bridge.stdout.readline())
            with (self.output / "remote_rpc.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "name": name,
                            "wall_s": time.monotonic() - started,
                            "ok": response.get("ok"),
                            "error": response.get("error"),
                        }
                    )
                    + "\n"
                )
            if not response["ok"]:
                raise RuntimeError(response["error"])
            return response["result"]

    def observe(self) -> Observation:
        payload = self._rpc("observe_policy")
        observation = decode_observation(payload)
        with (self.output / "local_observations.jsonl").open("a") as stream:
            stream.write(json.dumps(asdict(observation)) + "\n")
        return observation

    def execute(self, action: Action) -> None:
        from .deployment import reject_base_command

        reject_base_command(action)
        answer = self._rpc("execute_action", {"action": asdict(action)})
        with (self.output / "local_actions.jsonl").open("a") as stream:
            stream.write(json.dumps({"action": asdict(action), "result": answer}) + "\n")

    def check_reachability(self, action: Action) -> bool:
        return self._rpc("check_reachability", {"action": asdict(action)}).get("reachable") is True

    def stop(self, reason: str) -> None:
        if not self._closed:
            self._rpc("stop", {"reason": reason})

    def native_success(self) -> bool:
        status = self._rpc("native_status")
        (self.output / "latest_native_status.json").write_text(json.dumps(status, indent=2) + "\n")
        return status.get("native_success") is True

    def env_steps(self) -> int:
        return int(self._rpc("native_status")["steps"])

    def artifact(self, name: str, step: int = -1) -> bytes:
        if name != "overview.png" or step != -1:
            raise ValueError("Only latest native camera images are available")
        target = self.output / "cameras"
        target.mkdir(exist_ok=True)
        self._rsync(f"{self.config.host}:{self.remote}/{name}", str(target / name))
        return (target / name).read_bytes()

    def sync_artifacts(self) -> None:
        self._rsync(f"{self.config.host}:{self.remote}/", str(self.output) + "/")
        self._rsync(
            f"{self.config.host}:{self.launch}/worker.log", str(self.output / "remote_worker.log")
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._rpc("close")
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                final = self._python(f"""import json
from pathlib import Path
p=Path({self.remote!r})
finished=p/'finished.json'
pid=Path({(self.launch + "/worker.pid")!r}).read_text().strip()
proc=Path('/proc')/pid/'stat'
alive=proc.exists() and proc.read_text().split(') ',1)[1][0]!='Z'
data=json.loads(finished.read_text()) if finished.exists() else {{}}
print(json.dumps({{'finalized':data.get('evidence_finalized') is True,'alive':alive}}))
""")
                if final["finalized"]:
                    break
                if not final["alive"]:
                    raise RuntimeError("Native worker exited before evidence finalization")
                time.sleep(0.2)
            else:
                raise TimeoutError("Native evidence finalization exceeded 30 seconds")
        except Exception:
            self._stop_own_worker()
            raise
        finally:
            self._closed = True
            if self._bridge is not None:
                self._bridge.stdin.close()
                try:
                    self._bridge.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._bridge.terminate()
            if self._bridge_errors:
                self._bridge_errors.close()
            self.sync_artifacts()
            render_sensor_video(self.output)


def render_sensor_video(output: Path) -> dict:
    """Encode already recorded native RGB frames; never replay robot commands."""
    png_frames = sorted((output / "frames").glob("frame_*.png"))
    frames = sorted((output / "sensors").glob("frame_*.npz"))
    result = {
        "kind": "recorded_native_sensor_diagnostic",
        "new_evaluation": False,
        "frames": len(png_frames) or len(frames),
        "video": None,
    }
    if png_frames and shutil.which("ffmpeg"):
        import statistics

        timestamps = output / "frames/timestamps.jsonl"
        stamps = [json.loads(line)["stamp"] for line in timestamps.read_text().splitlines()]
        intervals = [b - a for a, b in zip(stamps, stamps[1:], strict=False) if b > a]
        fps = 1 / statistics.median(intervals) if intervals else 25.0
        path = output / "native_observations.mp4"
        if path.exists():
            raise FileExistsError(path)
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-pattern_type",
                "glob",
                "-i",
                str(output / "frames/frame_*.png"),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
            timeout=120,
        )
        result.update(
            video=path.name,
            video_sha256=file_hash(path),
            fps=fps,
            timestamps_sha256=file_hash(timestamps),
            timing="Native camera sequence; rate inferred from recorded simulation timestamps",
        )
    elif frames and shutil.which("ffmpeg"):
        import numpy as np

        first = np.load(frames[0])["rgb"][:, :, :3]
        height, width = first.shape[:2]
        path = output / "native_observations.mp4"
        if path.exists():
            raise FileExistsError(path)
        command = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            "10",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for frame in frames:
                with np.load(frame) as data:
                    rgb = np.asarray(data["rgb"][:, :, :3], dtype=np.uint8)
                    if rgb.shape != first.shape:
                        raise ValueError("Native camera frame dimensions changed")
                    process.stdin.write(rgb.tobytes())
        finally:
            process.stdin.close()
        if process.wait(timeout=60) != 0:
            raise RuntimeError("Native diagnostic video encoding failed")
        result.update(
            video=path.name,
            video_sha256=file_hash(path),
            fps=10,
            timing="Observation sequence at 10 fps; not real-time duration",
        )
    else:
        result["unavailable"] = "No recorded RGB frames or ffmpeg unavailable"
    write_new_json(output / "video_provenance.json", result)
    return result


def make_backend(
    *, out_dir: Path, scenario: dict, parameters: dict, binding: dict
) -> RemoteNativeBackend:
    """Factory consumed by airport.checkpoint.agent; launch one fresh native worker."""
    backend = RemoteNativeBackend(out_dir, binding, RemoteConfig.from_environment())
    backend.prepare_and_start()
    return backend
