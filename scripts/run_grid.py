import argparse
import yaml
import os
import json
import copy

from engine.run_walkforward import run_walkforward  # 기존 엔진 호출 부분 유지


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_selection(base_config, portfolio_config):
    """
    base_config의 selection을 portfolio_config로 덮어쓴다.
    """
    if not portfolio_config:
        return base_config

    merged = copy.deepcopy(base_config)

    if "selection" not in merged:
        merged["selection"] = {}

    for k, v in portfolio_config.get("selection", {}).items():
        merged["selection"][k] = v

    return merged


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prices", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--portfolio-config", required=False)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    base_config = load_yaml(args.config)
    grid_config = load_yaml(args.grid)

    portfolio_config = None
    if args.portfolio_config:
        portfolio_config = load_yaml(args.portfolio_config)

    # 🔥 selection 병합
    final_config = merge_selection(base_config, portfolio_config)

    # 🔥 walkforward 실행
    results = run_walkforward(
        prices_path=args.prices,
        base_config=final_config,
        grid_config=grid_config,
        out_dir=args.out_dir,
    )

    # 결과 저장 (기존 구조 유지)
    with open(os.path.join(args.out_dir, "best_params.json"), "w") as f:
        json.dump(results["best_params"], f, indent=2)

    print("Grid run complete.")


if __name__ == "__main__":
    main()