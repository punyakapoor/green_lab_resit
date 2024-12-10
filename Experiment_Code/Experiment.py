import subprocess
import time
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from ProgressManager.Output.OutputProcedure import OutputProcedure as output
from typing import Dict, Any, Optional
import re
import itertools
import deepeval  
import logging
import os
import torch
import gc
import psutil


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('experiment_errors.log')
fh.setLevel(logging.ERROR)
fh.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
logger.addHandler(ch)
os.environ['TRANSFORMERS_CACHE'] = '/home/greenhallucinations/.cache/huggingface_custom'
os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)

class RunnerConfig:
    ROOT_DIR = Path(__file__).resolve().parent
    name: str = "llm_prompt_energy_study"
    results_output_path: Path = ROOT_DIR / 'experiments'
    operation_type: OperationType = OperationType.AUTO
    time_between_runs_in_ms: int = 1000

    def __init__(self):
        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
            (RunnerEvents.BEFORE_RUN, self.before_run),
            (RunnerEvents.START_RUN, self.start_run),
            (RunnerEvents.START_MEASUREMENT, self.start_measurement),
            (RunnerEvents.INTERACT, self.interact),
            (RunnerEvents.STOP_MEASUREMENT, self.stop_measurement),
            (RunnerEvents.STOP_RUN, self.stop_run),
            (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
            (RunnerEvents.AFTER_EXPERIMENT, self.after_experiment),
        ])
        self.run_table_model = None
        self.factors = []
        self.energy_consumption = None   
        self.evaluation_results = None
        self.response_time = None
        self.generated_response = None
        self.powerstat_output = None
        self.model_cache = {}
        self.gpu_utilization = None
        self.gpu_power = None
        self.vram_usage = None
        self.cpu_utilization = None

        output.console_log("Custom config loaded")
        print("DEBUG: RunnerConfig initialized!")

    def create_run_table_model(self) -> RunTableModel:
        factor1 = FactorModel("model", ["Qwen_v2.5", "Gemma_v2", "Mistral_v0.3"])
        factor2 = FactorModel("prompt_type", ["classification", "sentiment_analysis", "summarization"])
        factor3 = FactorModel("input_length", ["short", "long"])

        self.factors = [factor1, factor2, factor3]

        self.run_table_model = RunTableModel(
            factors=self.factors,
            exclude_variations=[],
            repetitions=3,
            data_columns=[
                "response_time", "energy_consumption", "generated_response", 
                "evaluation_results", "gpu_utilization", "gpu_power",
                "vram_usage", "cpu_utilization", "cpu_power"
            ]
        )
        print("DEBUG: Run table model created!")
        return self.run_table_model

    def generate_all_variations(self):
        factor_levels = [f.treatments for f in self.factors]
        all_combinations = list(itertools.product(*factor_levels))
        factor_names = [f.factor_name for f in self.factors]

        variations = []
        for combo in all_combinations:
            variation_dict = dict(zip(factor_names, combo))
            for _ in range(self.run_table_model.repetitions):
                variations.append(variation_dict.copy())
        print(f"DEBUG: Generated {len(variations)} variations.")
        return variations

    def run_powerstat_with_password(self):
        password = os.getenv('POWERSTAT_PASSWORD')  
        if not password:
            error_msg = "ERROR: PowerStat password not set in environment variable 'POWERSTAT_PASSWORD'."
            print(error_msg)
            logging.error(error_msg)
            return None
        command = ["sudo", "-S", "powerstat", "-d", "0", "-c", "1", "-R"]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            output_data, error = process.communicate(input=f"{password}\n".encode())
            print("DEBUG: Full PowerStat Output:\n", output_data.decode())
            print("DEBUG: PowerStat Errors:\n", error.decode())
            if process.returncode != 0:
                error_msg = f"ERROR: PowerStat failed to start. {error.decode()}"
                print(error_msg)
                logging.error(error_msg)
                return None
            return output_data.decode()
        except Exception as e:
            error_msg = f"ERROR: Failed to run PowerStat: {e}"
            print(error_msg)
            logging.error(error_msg)
            return None

    def parse_powerstat_output(self, powerstat_output: str) -> Optional[float]:
        try:
            for line in powerstat_output.splitlines():
                if line.strip().startswith("Average"):
                    print(f"DEBUG: Extracted Average Line: {line}")
                    floats = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                    if floats:
                        avg_value = float(floats[-1])
                        return avg_value
        except Exception as e:
            error_msg = f"ERROR: Failed to parse PowerStat output: {e}"
            print(error_msg)
            logging.error(error_msg)
        return None

    def get_gpu_metrics(self) -> (Optional[float], Optional[float], Optional[float]):
 
        command = [
            "nvidia-smi", 
            "--query-gpu=utilization.gpu,power.draw,memory.used",
            "--format=csv,noheader,nounits"
        ]
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output_data, error = process.communicate()
            if process.returncode != 0:
                err_msg = f"ERROR: Failed to run nvidia-smi: {error.decode()}"
                print(err_msg)
                logging.error(err_msg)
                return None, None, None

            data = output_data.decode().strip().split('\n')
            if len(data) < 1:
                return None, None, None
            line = data[0].strip()
            vals = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            if len(vals) >= 3:
                gpu_util = float(vals[0])
                gpu_power = float(vals[1])
                vram_usage = float(vals[2])
                return gpu_util, gpu_power, vram_usage
        except Exception as e:
            err_msg = f"ERROR: Exception while getting GPU metrics: {e}"
            print(err_msg)
            logging.error(err_msg)
        return None, None, None

    def before_experiment(self) -> None:
        output.console_log("Experiment started!")
        print("DEBUG: Experiment started!")

    def before_run(self) -> None:
        output.console_log("Preparing for the next run...")
        print("DEBUG: Before run!")

    def start_run(self, context: RunnerContext) -> None:
        output.console_log("Starting run...")
        print(f"DEBUG: Starting run {context.run_nr}")

    def start_measurement(self, context: RunnerContext) -> None:
        output.console_log("Starting measurement...")
        print(f"DEBUG: Starting measurement for run {context.run_nr}")
        self.powerstat_output = self.run_powerstat_with_password()

    def interact(self, context: RunnerContext) -> None:
        output.console_log("Interacting with the system...")
        print(f"DEBUG: Interacting with the system for run {context.run_nr}")

        model_paths = {
            "Qwen_v2.5": "Qwen/Qwen2.5-7B",
            "Gemma_v2": "google/gemma-2b",
            "Mistral_v0.3": "mistralai/Mistral-7B-Instruct-v0.3"
        }

        model_name = context.run_variation["model"]
        prompt_type = context.run_variation["prompt_type"]
        input_length = context.run_variation["input_length"]

        if model_name in self.model_cache:
            model, tokenizer = self.model_cache[model_name]
            print(f"DEBUG: Reusing cached model {model_name}.")
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_paths[model_name],
                    token=os.getenv('HUGGINGFACE_TOKEN')
                )

                if model_name == "Mistral_v0.3":
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                        print("DEBUG: Mistral pad_token set to eos_token.")

                model = AutoModelForCausalLM.from_pretrained(
                    model_paths[model_name],
                    token=os.getenv('HUGGINGFACE_TOKEN'),
                    device_map='auto',
                    torch_dtype=torch.float16,
                    offload_buffers=True
                )

                self.model_cache[model_name] = (model, tokenizer)
                print(f"DEBUG: Successfully loaded and cached model {model_name}.")
            except Exception as e:
                error_msg = f"ERROR: Could not load model {model_name}: {e}"
                print(error_msg)
                logging.error(error_msg)
                self.response_time = None
                self.generated_response = "Model loading error"
                self.evaluation_results = None
                return

        prompts = {
            "classification": "Is this text positive or negative?",
            "sentiment_analysis": "Analyze the sentiment of this text.",
            "summarization": "Summarize the following text."
        }
        prompt = prompts.get(prompt_type, "Summarize the following text.")

        try:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        except Exception as e:
            error_msg = f"ERROR: Tokenizer error for model {model_name} with prompt '{prompt}': {e}"
            print(error_msg)
            logging.error(error_msg)
            self.response_time = None
            self.generated_response = "Tokenizer error"
            self.evaluation_results = None
            return

        try:
            start_time = time.time()
            with torch.no_grad():
                with torch.autocast(model.device.type):
                    response = model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        pad_token_id=tokenizer.eos_token_id if model_name != "Mistral_v0.3" else tokenizer.pad_token_id,
                        max_length=50,
                        num_return_sequences=1,
                        do_sample=False,
                        early_stopping=True
                    )
            end_time = time.time()
            self.response_time = end_time - start_time
            print(f"DEBUG: Response time for run {context.run_nr}: {self.response_time} seconds")
        except torch.cuda.OutOfMemoryError:
            error_msg = f"ERROR: Out of memory for model {model_name} on run {context.run_nr}."
            print(error_msg)
            logging.error(error_msg)
            self.response_time = None
            self.generated_response = "OOM error"
            self.evaluation_results = None
            torch.cuda.empty_cache()
            gc.collect()
            return
        except Exception as e:
            error_msg = f"ERROR: Generation error for model {model_name}: {e}"
            print(error_msg)
            logging.error(error_msg)
            self.response_time = None
            self.generated_response = "Generation error"
            self.evaluation_results = None
            return

        try:
            self.generated_response = tokenizer.decode(response[0], skip_special_tokens=True)
            print(f"DEBUG: Generated response for run {context.run_nr}: {self.generated_response}")
        except Exception as e:
            error_msg = f"ERROR: Decoding error for run {context.run_nr}: {e}"
            print(error_msg)
            logging.error(error_msg)
            self.generated_response = "Decoding error"

        try:
            evaluator = deepeval.Evaluator()
            self.evaluation_results = evaluator.evaluate(prompt_type, self.generated_response)
            print(f"DEBUG: Evaluation Results for run {context.run_nr}: {self.evaluation_results}")
        except AttributeError as ae:
            error_msg = f"ERROR: DeepEval error for run {context.run_nr}: {ae}"
            print(error_msg)
            logging.error(error_msg)
            self.evaluation_results = None
        except Exception as e:
            error_msg = f"ERROR: DeepEval error for run {context.run_nr}: {e}"
            print(error_msg)
            logging.error(error_msg)
            self.evaluation_results = None

        torch.cuda.empty_cache()
        gc.collect()

    def stop_measurement(self, context: RunnerContext) -> None:
        output.console_log("Stopping measurement...")
        print(f"DEBUG: Stopping measurement for run {context.run_nr}")
        if self.powerstat_output:
            self.energy_consumption = self.parse_powerstat_output(self.powerstat_output)
            if self.energy_consumption is not None:
                print(f"DEBUG: CPU Power for run {context.run_nr}: {self.energy_consumption} Watts")
            else:
                warning_msg = f"WARNING: CPU power could not be determined for run {context.run_nr}."
                print(warning_msg)
                logging.warning(warning_msg)
        else:
            warning_msg = f"WARNING: No PowerStat output captured for run {context.run_nr}."
            print(warning_msg)
            logging.warning(warning_msg)

        self.cpu_utilization = psutil.cpu_percent(interval=None)
        print(f"DEBUG: CPU Utilization for run {context.run_nr}: {self.cpu_utilization}%")

        self.gpu_utilization, self.gpu_power, self.vram_usage = self.get_gpu_metrics()
        if self.gpu_utilization is not None:
            print(f"DEBUG: GPU Utilization for run {context.run_nr}: {self.gpu_utilization}%")
        else:
            print("DEBUG: Could not get GPU Utilization")

        if self.gpu_power is not None:
            print(f"DEBUG: GPU Power for run {context.run_nr}: {self.gpu_power} W")

        if self.vram_usage is not None:
            print(f"DEBUG: VRAM Usage for run {context.run_nr}: {self.vram_usage} MB")

    def stop_run(self, context: RunnerContext) -> None:
        output.console_log("Run complete!")
        print(f"DEBUG: Run {context.run_nr} complete!")

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, Any]]:
        output.console_log("Populating run data...")
        print(f"DEBUG: Populating run data for run {context.run_nr}")

        run_data = {
            "run_nr": context.run_nr,
            "model": context.run_variation["model"],
            "prompt_type": context.run_variation["prompt_type"],
            "input_length": context.run_variation["input_length"],
            "response_time": self.response_time,
            "cpu_power": self.energy_consumption,
            "generated_response": self.generated_response,
            "evaluation_results": self.evaluation_results,
            "gpu_utilization": self.gpu_utilization,
            "gpu_power": self.gpu_power,
            "vram_usage": self.vram_usage,
            "cpu_utilization": self.cpu_utilization
        }

        results_file = self.results_output_path / "experiment_results.json"
        try:
            with open(results_file, "a") as f:
                f.write(json.dumps(run_data) + "\n")
            print(f"DEBUG: Run {context.run_nr} data recorded.")
        except Exception as e:
            error_msg = f"ERROR: Failed to write run {context.run_nr} data: {e}"
            print(error_msg)
            logging.error(error_msg)
            return None

        return run_data

    def after_experiment(self) -> None:
        output.console_log("Experiment completed!")
        print("DEBUG: Experiment completed!")


def run_single_variation(run_id, variation, config):
    run_dir = config.results_output_path / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    context = RunnerContext(run_variation=variation, run_nr=run_id, run_dir=run_dir)

    config.before_run()
    config.start_run(context)
    config.start_measurement(context)
    config.interact(context)
    config.stop_measurement(context)
    config.stop_run(context)
    run_data = config.populate_run_data(context)
    print(f"DEBUG: Run {run_id} Data: {run_data}")


def main():
    print("DEBUG: Script execution started!")
    config = RunnerConfig()
    config.create_run_table_model()

    all_run_variations = config.generate_all_variations()
    print(f"DEBUG: Total Measurements to Run: {len(all_run_variations)}")

    config.before_experiment()
    run_id = 0

    for variation in all_run_variations:
        run_id += 1
        run_single_variation(run_id, variation, config)

    config.after_experiment()

if __name__ == "__main__":
    main()