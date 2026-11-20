"""Short Kit smoke test: launch, reset, step a few times, exit."""

from __future__ import annotations

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Verify Isaac Lab Kit launch.")
add_launcher_args(parser)
args_cli = parser.parse_args()

from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402


def main() -> None:
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
        sim.reset()
        print("[INFO]: Setup complete...")
        for _ in range(10):
            sim.step()
        print("[INFO]: Verify steps finished; exiting.")


if __name__ == "__main__":
    main()
