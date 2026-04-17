# -*- coding: utf-8 -*-

import argparse
import ast
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional


ToolHandler = Callable[[Dict[str, Any]], Dict[str, Any]]


def _as_path(path):
    return os.fspath(path) if isinstance(path, os.PathLike) else path


def _read_h5ad(path):
    import anndata as ad

    return ad.read_h5ad(_as_path(path))


def _bool_value(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _maybe_number(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return value
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return value


def _parse_key_value(items):
    params = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError("Parameter '%s' must use key=value format." % item)
        key, value = item.split("=", 1)
        key = key.strip().replace("-", "_")
        value = value.strip()
        try:
            params[key] = ast.literal_eval(value)
        except Exception:
            params[key] = _maybe_number(value)
    return params


def _parse_point(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if text.startswith("[") or text.startswith("("):
        return list(ast.literal_eval(text))
    return [_maybe_number(part.strip()) for part in text.split(",")]


@dataclass
class ToolSpec:
    name: str
    description: str
    required: Iterable[str]
    optional: Iterable[str] = field(default_factory=tuple)
    handler: Optional[ToolHandler] = None

    def missing(self, params):
        return [key for key in self.required if params.get(key) in (None, "")]

    def schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "required": list(self.required),
            "optional": list(self.optional),
        }


@dataclass
class AgentPlan:
    task: str
    tool: Optional[str]
    params: Dict[str, Any]
    missing: List[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self):
        return {
            "task": self.task,
            "tool": self.tool,
            "params": self.params,
            "missing": self.missing,
            "rationale": self.rationale,
        }


class GeneRhythmAgent:
    """
    Lightweight agent layer for GeneRhythm.

    The agent maps a natural-language task plus structured params to a registered
    tool, validates required arguments, and optionally executes the tool. A real
    LLM planner can be plugged in by passing ``planner(task, params, tools)``.
    """

    def __init__(self, planner=None):
        self.planner = planner
        self.tools = self._build_tools()

    def _build_tools(self):
        specs = [
            ToolSpec(
                name="frequency_format",
                description="Import/format a trajectory table and calculate frequency features.",
                required=("trajectory", "adata", "dataset"),
                optional=("time_key", "path_key", "path_id", "expression_layer", "n_bins", "trajectory_csv", "output_npy"),
                handler=self._tool_frequency_format,
            ),
            ToolSpec(
                name="frequency_scvelo",
                description="Run scVelo trajectory inference and calculate frequency features.",
                required=("adata", "dataset"),
                optional=("mode", "time_key", "path_key", "path_id", "n_bins", "output_npy", "trajectory_csv"),
                handler=self._tool_frequency_scvelo,
            ),
            ToolSpec(
                name="frequency_celldancer",
                description="Run cellDancer trajectory inference and calculate frequency features.",
                required=("dataset",),
                optional=("adata", "table", "gene_list", "time_key", "path_key", "path_id", "n_bins", "output_npy", "trajectory_csv"),
                handler=self._tool_frequency_celldancer,
            ),
            ToolSpec(
                name="frequency_veloagent",
                description="Run VeloAgent trajectory inference and calculate frequency features.",
                required=("adata", "dataset"),
                optional=("time_key", "path_key", "path_id", "n_bins", "output_npy", "trajectory_csv"),
                handler=self._tool_frequency_veloagent,
            ),
            ToolSpec(
                name="frequency_spatial",
                description="Calculate spatial trajectory frequency features from a start/end line.",
                required=("adata", "dataset", "start", "end"),
                optional=(),
                handler=self._tool_frequency_spatial,
            ),
            ToolSpec(
                name="train_gene_rhythm",
                description="Train the GCN/VAE model on extracted frequency features.",
                required=("input_data",),
                optional=("graph", "model_output", "latent_output", "sc_data", "lr", "n_epoch", "batch_size"),
                handler=self._tool_train_gene_rhythm,
            ),
        ]
        return {spec.name: spec for spec in specs}

    def available_tools(self):
        return [tool.schema() for tool in self.tools.values()]

    def plan(self, task, params=None):
        params = dict(params or {})
        task_text = " ".join(task) if isinstance(task, (list, tuple)) else str(task)

        if self.planner is not None:
            planned = self.planner(task_text, params, self.available_tools())
            tool_name = planned.get("tool")
            plan_params = dict(params)
            plan_params.update(planned.get("params", {}))
            return self._make_plan(task_text, tool_name, plan_params, planned.get("rationale", "LLM planner"))

        lowered = task_text.lower()
        if "celldancer" in lowered or "cell dancer" in lowered:
            tool_name = "frequency_celldancer"
            rationale = "cellDancer keyword matched."
        elif "scvelo" in lowered or "sc velo" in lowered:
            tool_name = "frequency_scvelo"
            rationale = "scVelo keyword matched."
        elif "veloagent" in lowered or "velo agent" in lowered:
            tool_name = "frequency_veloagent"
            rationale = "VeloAgent keyword matched."
        elif "spatial" in lowered or "space" in lowered:
            tool_name = "frequency_spatial"
            rationale = "spatial keyword matched."
        elif "train" in lowered or "vae" in lowered or "gcn" in lowered:
            tool_name = "train_gene_rhythm"
            rationale = "training keyword matched."
        elif "format" in lowered or "trajectory" in lowered or "frequency" in lowered:
            tool_name = "frequency_format"
            rationale = "trajectory/frequency keyword matched."
        else:
            return AgentPlan(
                task=task_text,
                tool=None,
                params=params,
                missing=[],
                rationale="No matching tool. Call available_tools() to inspect options.",
            )

        return self._make_plan(task_text, tool_name, params, rationale)

    def _make_plan(self, task, tool_name, params, rationale):
        if tool_name not in self.tools:
            return AgentPlan(
                task=task,
                tool=None,
                params=params,
                missing=[],
                rationale="Unknown tool '%s'." % tool_name,
            )
        tool = self.tools[tool_name]
        return AgentPlan(
            task=task,
            tool=tool_name,
            params=params,
            missing=tool.missing(params),
            rationale=rationale,
        )

    def run(self, task, params=None, execute=True):
        plan = self.plan(task, params=params)
        if plan.tool is None or plan.missing or not execute:
            return {"plan": plan.to_dict(), "executed": False}

        tool = self.tools[plan.tool]
        result = tool.handler(plan.params)
        return {"plan": plan.to_dict(), "executed": True, "result": result}

    def _tool_frequency_format(self, params):
        from Frequency_extract import frequency_extract, frequency_extract_format

        adata = _read_h5ad(params["adata"])
        dataset = params["dataset"]
        trajectory_info = frequency_extract_format(
            params["trajectory"],
            adata=adata,
            dataset=dataset,
            time_key=params.get("time_key"),
            path_key=params.get("path_key"),
            expression_layer=params.get("expression_layer"),
            n_bins=int(params.get("n_bins", 20)),
            save_csv=True,
            output_csv=params.get("trajectory_csv"),
        )
        output_npy = params.get("output_npy") or "%s.npy" % dataset
        frequency_extract(
            trajectory_info,
            adata,
            dataset,
            path_id=params.get("path_id", 1),
            output_npy=output_npy,
        )
        return {"output_npy": output_npy, "trajectory_csv": params.get("trajectory_csv")}

    def _tool_frequency_scvelo(self, params):
        from Frequency_extract import frequency_extract_scvelo

        dataset = params["dataset"]
        output_npy = params.get("output_npy") or "%s.npy" % dataset
        frequency_extract_scvelo(
            _read_h5ad(params["adata"]),
            dataset,
            mode=params.get("mode", "dynamical"),
            time_key=params.get("time_key"),
            path_key=params.get("path_key"),
            path_id=params.get("path_id", 1),
            n_bins=int(params.get("n_bins", 20)),
            trajectory_csv=params.get("trajectory_csv"),
            output_npy=output_npy,
        )
        return {"output_npy": output_npy, "trajectory_csv": params.get("trajectory_csv")}

    def _tool_frequency_celldancer(self, params):
        from Frequency_extract import frequency_extract_celldancer

        if not params.get("adata") and not params.get("table"):
            raise ValueError("frequency_celldancer requires either adata=... or table=...")

        dataset = params["dataset"]
        output_npy = params.get("output_npy") or "%s.npy" % dataset
        adata = _read_h5ad(params["adata"]) if params.get("adata") else None
        adata_or_df = params["table"] if params.get("table") else adata
        frequency_extract_celldancer(
            adata_or_df,
            dataset,
            adata=adata,
            gene_list=params.get("gene_list"),
            time_key=params.get("time_key", "pseudotime"),
            path_key=params.get("path_key"),
            path_id=params.get("path_id", 1),
            n_bins=int(params.get("n_bins", 20)),
            trajectory_csv=params.get("trajectory_csv"),
            output_npy=output_npy,
        )
        return {"output_npy": output_npy, "trajectory_csv": params.get("trajectory_csv")}

    def _tool_frequency_veloagent(self, params):
        from Frequency_extract import frequency_extract_veloagent

        dataset = params["dataset"]
        output_npy = params.get("output_npy") or "%s.npy" % dataset
        frequency_extract_veloagent(
            _read_h5ad(params["adata"]),
            dataset,
            time_key=params.get("time_key"),
            path_key=params.get("path_key"),
            path_id=params.get("path_id", 1),
            n_bins=int(params.get("n_bins", 20)),
            trajectory_csv=params.get("trajectory_csv"),
            output_npy=output_npy,
        )
        return {"output_npy": output_npy, "trajectory_csv": params.get("trajectory_csv")}

    def _tool_frequency_spatial(self, params):
        from Frequency_extract import frequency_extract_spatial

        dataset = params["dataset"]
        output_npy = "%s.npy" % dataset
        frequency_extract_spatial(
            _read_h5ad(params["adata"]),
            dataset,
            start=_parse_point(params["start"]),
            end=_parse_point(params["end"]),
        )
        return {"output_npy": output_npy}

    def _tool_train_gene_rhythm(self, params):
        from GCN_VAE import GeneRhythm_Model

        model_output = params.get("model_output", "gcn_vae.pth")
        latent_output = params.get("latent_output", "ALL_mu.npy")
        GeneRhythm_Model(
            params["input_data"],
            graph=params.get("graph"),
            model_output=model_output,
            latent_output=latent_output,
            sc_data=params.get("sc_data"),
            lr=float(params.get("lr", 0.0005)),
            n_epoch=int(params.get("n_epoch", 1000)),
            batch_size=int(params.get("batch_size", 32)),
        )
        return {"model_output": model_output, "latent_output": latent_output}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Agentic CLI for GeneRhythm workflows.")
    parser.add_argument("task", nargs="*", help="Natural-language task, e.g. 'run scvelo frequency'.")
    parser.add_argument("--param", action="append", default=[], help="Extra key=value parameter. Can repeat.")
    parser.add_argument("--adata")
    parser.add_argument("--trajectory")
    parser.add_argument("--table")
    parser.add_argument("--dataset")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--input-data", dest="input_data")
    parser.add_argument("--graph")
    parser.add_argument("--output-npy", dest="output_npy")
    parser.add_argument("--trajectory-csv", dest="trajectory_csv")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--tools", action="store_true", help="Print registered tools and exit.")
    args = parser.parse_args(argv)

    agent = GeneRhythmAgent()
    if args.tools:
        print(json.dumps(agent.available_tools(), indent=2, ensure_ascii=False))
        return

    params = _parse_key_value(args.param)
    for key in (
        "adata",
        "trajectory",
        "table",
        "dataset",
        "start",
        "end",
        "input_data",
        "graph",
        "output_npy",
        "trajectory_csv",
    ):
        value = getattr(args, key)
        if value is not None:
            params[key] = value

    task = " ".join(args.task) if args.task else "frequency"
    output = agent.run(task, params=params, execute=not args.plan_only)
    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()


# Backward-compatible alias for early drafts that used the wrong project name.
ScGeneRhythmAgent = GeneRhythmAgent
