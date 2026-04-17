# -*- coding: utf-8 -*-

import argparse
import ast
import json
import os
import traceback
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


def _safe_json(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


@dataclass
class ToolSpec:
    name: str
    description: str
    required: Iterable[str]
    optional: Iterable[str] = field(default_factory=tuple)
    handler: Optional[ToolHandler] = None

    def missing(self, params: Dict[str, Any]) -> List[str]:
        return [key for key in self.required if params.get(key) in (None, "")]

    def schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "required": list(self.required),
            "optional": list(self.optional),
        }


@dataclass
class WorkflowStep:
    tool: str
    params: Dict[str, Any]
    missing: List[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self):
        return {
            "tool": self.tool,
            "params": self.params,
            "missing": self.missing,
            "rationale": self.rationale,
        }


@dataclass
class WorkflowPlan:
    task: str
    steps: List[WorkflowStep] = field(default_factory=list)
    executable: bool = False
    rationale: str = ""

    def to_dict(self):
        return {
            "task": self.task,
            "steps": [step.to_dict() for step in self.steps],
            "executable": self.executable,
            "rationale": self.rationale,
        }


@dataclass
class AgentState:
    dataset: Optional[str] = None
    adata: Optional[str] = None
    trajectory: Optional[str] = None
    table: Optional[str] = None
    graph: Optional[str] = None
    input_data: Optional[str] = None
    trajectory_csv: Optional[str] = None
    output_npy: Optional[str] = None
    latent_output: Optional[str] = None
    model_output: Optional[str] = None
    last_tool: Optional[str] = None

    def to_dict(self):
        return self.__dict__.copy()

    def update_from_params(self, params: Dict[str, Any]):
        for key in self.__dict__.keys():
            if key in params and params[key] not in (None, ""):
                setattr(self, key, params[key])

    def update_from_result(self, result: Dict[str, Any], tool_name: Optional[str] = None):
        if tool_name:
            self.last_tool = tool_name
        for key in self.__dict__.keys():
            if key in result and result[key] not in (None, ""):
                setattr(self, key, result[key])

        # Helpful carry-forward rule:
        # if output_npy exists, it can often be reused as input_data for training.
        if result.get("output_npy"):
            self.input_data = result["output_npy"]


class GeneRhythmAgent:
    """
    Multi-step lightweight agent for GeneRhythm workflows.

    Features
    --------
    1. Route natural-language tasks to registered tools.
    2. Support multi-step workflow plans.
    3. Keep session state across steps.
    4. Allow dry-run planning without execution.
    5. Provide actionable suggestions when parameters are missing.
    6. Preserve a hook for plugging in an external LLM planner.
    """

    def __init__(self, planner=None):
        self.planner = planner
        self.tools = self._build_tools()
        self.state = AgentState()

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
                optional=("output_npy",),
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

    def reset_state(self):
        self.state = AgentState()

    def _merge_with_state(self, params: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
        merged = dict(params)
        for key in keys:
            if merged.get(key) in (None, ""):
                state_value = getattr(self.state, key, None)
                if state_value not in (None, ""):
                    merged[key] = state_value
        return merged

    def _make_step(self, tool_name: str, params: Dict[str, Any], rationale: str) -> WorkflowStep:
        if tool_name not in self.tools:
            return WorkflowStep(
                tool=tool_name,
                params=params,
                missing=[],
                rationale="Unknown tool '%s'." % tool_name,
            )
        tool = self.tools[tool_name]
        return WorkflowStep(
            tool=tool_name,
            params=params,
            missing=tool.missing(params),
            rationale=rationale,
        )

    def plan(self, task, params=None) -> WorkflowPlan:
        params = dict(params or {})
        task_text = " ".join(task) if isinstance(task, (list, tuple)) else str(task)
        lowered = task_text.lower().strip()

        # Update state with explicitly provided params first.
        self.state.update_from_params(params)

        if self.planner is not None:
            planned = self.planner(task_text, params, self.available_tools(), self.state.to_dict())
            step_defs = planned.get("steps", [])
            if not step_defs and planned.get("tool"):
                step_defs = [{
                    "tool": planned["tool"],
                    "params": planned.get("params", {}),
                    "rationale": planned.get("rationale", "LLM planner"),
                }]

            steps = []
            for step_def in step_defs:
                tool_name = step_def.get("tool")
                step_params = dict(params)
                step_params.update(step_def.get("params", {}))
                steps.append(self._make_step(tool_name, step_params, step_def.get("rationale", "LLM planner")))

            executable = len(steps) > 0 and all(len(step.missing) == 0 for step in steps)
            return WorkflowPlan(
                task=task_text,
                steps=steps,
                executable=executable,
                rationale=planned.get("rationale", "LLM planner"),
            )

        steps: List[WorkflowStep] = []
        rationale_parts: List[str] = []

        wants_train = any(k in lowered for k in ("train", "vae", "gcn", "model"))
        wants_format = any(k in lowered for k in ("format", "trajectory table", "trajectory csv"))
        wants_frequency = any(k in lowered for k in ("frequency", "extract", "feature"))
        wants_scvelo = any(k in lowered for k in ("scvelo", "sc velo"))
        wants_celldancer = any(k in lowered for k in ("celldancer", "cell dancer"))
        wants_veloagent = any(k in lowered for k in ("veloagent", "velo agent"))
        wants_spatial = any(k in lowered for k in ("spatial", "space"))

        # Compare workflow
        if "compare" in lowered and (wants_scvelo or wants_celldancer or wants_veloagent):
            if wants_scvelo:
                step_params = self._merge_with_state(params, ["adata", "dataset", "trajectory_csv", "output_npy"])
                if "output_npy" not in step_params or not step_params["output_npy"]:
                    if step_params.get("dataset"):
                        step_params["output_npy"] = "%s_scvelo.npy" % step_params["dataset"]
                steps.append(self._make_step("frequency_scvelo", step_params, "Comparison requested: added scVelo extraction."))
                rationale_parts.append("Added scVelo branch for comparison.")

            if wants_celldancer:
                step_params = self._merge_with_state(params, ["adata", "table", "dataset", "trajectory_csv", "output_npy"])
                if "output_npy" not in step_params or not step_params["output_npy"]:
                    if step_params.get("dataset"):
                        step_params["output_npy"] = "%s_celldancer.npy" % step_params["dataset"]
                steps.append(self._make_step("frequency_celldancer", step_params, "Comparison requested: added cellDancer extraction."))
                rationale_parts.append("Added cellDancer branch for comparison.")

            if wants_veloagent:
                step_params = self._merge_with_state(params, ["adata", "dataset", "trajectory_csv", "output_npy"])
                if "output_npy" not in step_params or not step_params["output_npy"]:
                    if step_params.get("dataset"):
                        step_params["output_npy"] = "%s_veloagent.npy" % step_params["dataset"]
                steps.append(self._make_step("frequency_veloagent", step_params, "Comparison requested: added VeloAgent extraction."))
                rationale_parts.append("Added VeloAgent branch for comparison.")

        else:
            # Single/multi-step workflow routing
            if wants_spatial:
                step_params = self._merge_with_state(params, ["adata", "dataset", "start", "end", "output_npy"])
                steps.append(self._make_step("frequency_spatial", step_params, "Spatial keyword matched."))
                rationale_parts.append("Added spatial frequency extraction step.")

            elif wants_scvelo:
                step_params = self._merge_with_state(params, ["adata", "dataset", "trajectory_csv", "output_npy"])
                steps.append(self._make_step("frequency_scvelo", step_params, "scVelo keyword matched."))
                rationale_parts.append("Added scVelo frequency extraction step.")

            elif wants_celldancer:
                step_params = self._merge_with_state(params, ["adata", "table", "dataset", "trajectory_csv", "output_npy"])
                steps.append(self._make_step("frequency_celldancer", step_params, "cellDancer keyword matched."))
                rationale_parts.append("Added cellDancer frequency extraction step.")

            elif wants_veloagent:
                step_params = self._merge_with_state(params, ["adata", "dataset", "trajectory_csv", "output_npy"])
                steps.append(self._make_step("frequency_veloagent", step_params, "VeloAgent keyword matched."))
                rationale_parts.append("Added VeloAgent frequency extraction step.")

            elif wants_format or wants_frequency:
                step_params = self._merge_with_state(params, ["trajectory", "adata", "dataset", "trajectory_csv", "output_npy"])
                steps.append(self._make_step("frequency_format", step_params, "Trajectory/frequency keyword matched."))
                rationale_parts.append("Added generic trajectory formatting and frequency extraction step.")

            if wants_train:
                step_params = self._merge_with_state(params, ["input_data", "graph", "model_output", "latent_output"])

                # Auto-handoff from previous output
                if step_params.get("input_data") in (None, "") and self.state.output_npy not in (None, ""):
                    step_params["input_data"] = self.state.output_npy

                # If training is requested together with a fresh extraction step, hand off expected output.
                if step_params.get("input_data") in (None, "") and steps:
                    previous_output = steps[-1].params.get("output_npy")
                    dataset_name = steps[-1].params.get("dataset") or params.get("dataset") or self.state.dataset
                    if previous_output:
                        step_params["input_data"] = previous_output
                    elif dataset_name:
                        step_params["input_data"] = "%s.npy" % dataset_name

                steps.append(self._make_step("train_gene_rhythm", step_params, "Training keyword matched."))
                rationale_parts.append("Added GeneRhythm training step.")

        executable = len(steps) > 0 and all(len(step.missing) == 0 for step in steps)

        if not steps:
            rationale = "No matching workflow. Call --tools to inspect available tools."
        else:
            rationale = " ".join(rationale_parts)

        return WorkflowPlan(
            task=task_text,
            steps=steps,
            executable=executable,
            rationale=rationale,
        )

    def _build_suggestions(self, plan: WorkflowPlan) -> List[str]:
        suggestions = []

        if not plan.steps:
            suggestions.append("No tool matched the task. Try including keywords like scvelo, celldancer, veloagent, spatial, format, or train.")
            suggestions.append("You can print registered tools with --tools.")
            return suggestions

        for step in plan.steps:
            if not step.missing:
                continue

            suggestions.append(
                "Tool '%s' is missing required parameters: %s"
                % (step.tool, ", ".join(step.missing))
            )

            if step.tool == "frequency_scvelo":
                suggestions.append("Provide --adata <file.h5ad> and --dataset <name>.")
            elif step.tool == "frequency_format":
                suggestions.append("Provide --trajectory <table/csv>, --adata <file.h5ad>, and --dataset <name>.")
            elif step.tool == "frequency_celldancer":
                suggestions.append("Provide --dataset <name> and either --adata <file.h5ad> or --table <trajectory_table>.")
            elif step.tool == "frequency_veloagent":
                suggestions.append("Provide --adata <file.h5ad> and --dataset <name>.")
            elif step.tool == "frequency_spatial":
                suggestions.append("Provide --adata <file.h5ad>, --dataset <name>, --start x,y, and --end x,y.")
            elif step.tool == "train_gene_rhythm":
                suggestions.append("Provide --input-data <features.npy>, or run a frequency extraction step first.")
        return suggestions

    def run(self, task, params=None, execute=True) -> Dict[str, Any]:
        params = dict(params or {})
        plan = self.plan(task, params=params)

        # Keep explicit user params in state
        self.state.update_from_params(params)

        if not execute or not plan.executable:
            return {
                "plan": plan.to_dict(),
                "executed": False,
                "state": self.state.to_dict(),
                "suggestions": self._build_suggestions(plan),
            }

        results = []
        for index, step in enumerate(plan.steps, start=1):
            tool = self.tools[step.tool]

            # Before each step, re-merge with latest state for missing optional carry-over
            step_params = self._merge_with_state(step.params, tool.required)
            step.params = step_params
            step.missing = tool.missing(step.params)

            if step.missing:
                results.append({
                    "step": index,
                    "tool": step.tool,
                    "ok": False,
                    "error": "Missing required parameters: %s" % ", ".join(step.missing),
                })
                return {
                    "plan": plan.to_dict(),
                    "executed": False,
                    "results": results,
                    "state": self.state.to_dict(),
                    "suggestions": self._build_suggestions(plan),
                }

            try:
                result = tool.handler(step.params)
                self.state.update_from_result(result, tool_name=step.tool)
                self.state.update_from_params(step.params)

                results.append({
                    "step": index,
                    "tool": step.tool,
                    "ok": True,
                    "params": step.params,
                    "result": result,
                })

            except Exception as e:
                results.append({
                    "step": index,
                    "tool": step.tool,
                    "ok": False,
                    "params": step.params,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                return {
                    "plan": plan.to_dict(),
                    "executed": False,
                    "results": results,
                    "state": self.state.to_dict(),
                    "suggestions": [
                        "Execution failed at step %d ('%s')." % (index, step.tool),
                        "Inspect the traceback field in the JSON output.",
                        "Check whether the required input files and imported modules are available.",
                    ],
                }

        return {
            "plan": plan.to_dict(),
            "executed": True,
            "results": results,
            "state": self.state.to_dict(),
        }

    # ----------------------------
    # Tool handlers
    # ----------------------------

    def _tool_frequency_format(self, params):
        from Frequency_extract import frequency_extract, frequency_extract_format

        adata = _read_h5ad(params["adata"])
        dataset = params["dataset"]

        trajectory_csv = params.get("trajectory_csv") or "%s_trajectory.csv" % dataset
        output_npy = params.get("output_npy") or "%s.npy" % dataset

        trajectory_info = frequency_extract_format(
            params["trajectory"],
            adata=adata,
            dataset=dataset,
            time_key=params.get("time_key"),
            path_key=params.get("path_key"),
            expression_layer=params.get("expression_layer"),
            n_bins=int(params.get("n_bins", 20)),
            save_csv=True,
            output_csv=trajectory_csv,
        )

        frequency_extract(
            trajectory_info,
            adata,
            dataset,
            path_id=params.get("path_id", 1),
            output_npy=output_npy,
        )

        return {
            "dataset": dataset,
            "adata": params["adata"],
            "trajectory": params["trajectory"],
            "trajectory_csv": trajectory_csv,
            "output_npy": output_npy,
        }

    def _tool_frequency_scvelo(self, params):
        from Frequency_extract import frequency_extract_scvelo

        dataset = params["dataset"]
        output_npy = params.get("output_npy") or "%s.npy" % dataset
        trajectory_csv = params.get("trajectory_csv") or "%s_scvelo_trajectory.csv" % dataset

        frequency_extract_scvelo(
            _read_h5ad(params["adata"]),
            dataset,
            mode=params.get("mode", "dynamical"),
            time_key=params.get("time_key"),
            path_key=params.get("path_key"),
            path_id=params.get("path_id", 1),
            n_bins=int(params.get("n_bins", 20)),
            trajectory_csv=trajectory_csv,
            output_npy=output_npy,
        )
        return {
            "dataset": dataset,
            "adata": params["adata"],
            "trajectory_csv": trajectory_csv,
            "output_npy": output_npy,
        }

    def _tool_frequency_celldancer(self, params):
        from Frequency_extract import frequency_extract_celldancer

        if not params.get("adata") and not params.get("table"):
            raise ValueError("frequency_celldancer requires either adata=... or table=...")

        dataset = params["dataset"]
        output_npy = params.get("output_npy") or "%s.npy" % dataset
        trajectory_csv = params.get("trajectory_csv") or "%s_celldancer_trajectory.csv" % dataset

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
            trajectory_csv=trajectory_csv,
            output_npy=output_npy,
        )
        return {
            "dataset": dataset,
            "adata": params.get("adata"),
            "table": params.get("table"),
            "trajectory_csv": trajectory_csv,
            "output_npy": output_npy,
        }

    def _tool_frequency_veloagent(self, params):
        from Frequency_extract import frequency_extract_veloagent

        dataset = params["dataset"]
        output_npy = params.get("output_npy") or "%s.npy" % dataset
        trajectory_csv = params.get("trajectory_csv") or "%s_veloagent_trajectory.csv" % dataset

        frequency_extract_veloagent(
            _read_h5ad(params["adata"]),
            dataset,
            time_key=params.get("time_key"),
            path_key=params.get("path_key"),
            path_id=params.get("path_id", 1),
            n_bins=int(params.get("n_bins", 20)),
            trajectory_csv=trajectory_csv,
            output_npy=output_npy,
        )
        return {
            "dataset": dataset,
            "adata": params["adata"],
            "trajectory_csv": trajectory_csv,
            "output_npy": output_npy,
        }

    def _tool_frequency_spatial(self, params):
        from Frequency_extract import frequency_extract_spatial

        dataset = params["dataset"]
        output_npy = params.get("output_npy") or "%s.npy" % dataset

        frequency_extract_spatial(
            _read_h5ad(params["adata"]),
            dataset,
            start=_parse_point(params["start"]),
            end=_parse_point(params["end"]),
        )
        return {
            "dataset": dataset,
            "adata": params["adata"],
            "output_npy": output_npy,
        }

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
        return {
            "input_data": params["input_data"],
            "graph": params.get("graph"),
            "model_output": model_output,
            "latent_output": latent_output,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Agentic CLI for GeneRhythm workflows (v2).")
    parser.add_argument("task", nargs="*", help="Natural-language task, e.g. 'run scvelo frequency then train'.")
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
    parser.add_argument("--latent-output", dest="latent_output")
    parser.add_argument("--model-output", dest="model_output")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--tools", action="store_true", help="Print registered tools and exit.")
    parser.add_argument("--reset-state", action="store_true", help="Reset in-memory state before running.")
    args = parser.parse_args(argv)

    agent = GeneRhythmAgent()

    if args.reset_state:
        agent.reset_state()

    if args.tools:
        print(_safe_json(agent.available_tools()))
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
        "latent_output",
        "model_output",
    ):
        value = getattr(args, key)
        if value is not None:
            params[key] = value

    task = " ".join(args.task) if args.task else "frequency"
    output = agent.run(task, params=params, execute=not args.plan_only)
    print(_safe_json(output))


if __name__ == "__main__":
    main()


