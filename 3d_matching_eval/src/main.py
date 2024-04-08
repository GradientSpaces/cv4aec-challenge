from collections import defaultdict
from email.policy import default
import json
import typer
import dotenv
import logging
import numpy as np

from pathlib import Path

import config
import loader
import matching
import utils

dotenv.load_dotenv()
utils.load_logger()


app = typer.Typer()


@app.command(
    name="match",
    help="Match two models"
)
def match(
    ref_data: Path = typer.Argument(..., help="Path to a ground-truth (reference) file or directory", exists=True, file_okay=True, dir_okay=True),
    user_data: Path = typer.Argument(..., help="Path to a target (user prediction) file or directory", exists=True, file_okay=True, dir_okay=True),
    output: Path = typer.Option("output", help="Path to the output directory", file_okay=False, dir_okay=True)
):
    output.mkdir(parents=True, exist_ok=True)
    logging.info("Reading the models..")

    ground_files = loader.read_source(ref_data.resolve())
    logging.info(f"Found {len(ground_files.keys())} GT models: {list(ground_files.keys())}")
    target_files = loader.read_source(user_data.resolve())
    logging.info(f"Found {len(target_files.keys())} Target models: {list(target_files.keys())}")

    global_metrics_output = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    global_iou_output = defaultdict(list)

    for model in target_files.keys():
        gtdata = ground_files.get(model)
        if gtdata is None:
            logging.error(f"Cannot find gt model for '{model}'.")
            continue
        tgdata = target_files[model]

        floors = {}
        for floor in tgdata.keys():
            tgfloor = tgdata[floor]
            gtfloor = gtdata.get(floor)
            if gtfloor is None:
                logging.error(f"Cannot find '{floor}' floor in gt model '{model}'.")
                continue
            else:
                logging.info(f"Matching '{model}', floor '{floor}'...")
                gtcounter = dict(zip(list(gtfloor.keys()), [len(values) for values in gtfloor.values()]))
                tgcounter = dict(zip(list(tgfloor.keys()), [len(values) for values in tgfloor.values()]))
                logging.info(f"Ground keys: {gtcounter}")
                logging.info(f"Target keys: {tgcounter}")

            gtstructures = loader.read_structures(gtfloor)
            tgstructures = loader.read_structures(tgfloor)

            floors[floor] = matching.match(gtstructures, tgstructures, output, model, floor)

        # NOTE: If any more data specific operations occurs use Pandas table.
        # Do not create any more defaultdicts...
        metrics_output = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        iou_output = defaultdict(list)
        for floor in floors.values():
            for classname, ious in floor["ious"]["general"].items():
                mean = ious["mean"]
                # Collect average IoU over claasses.
                global_iou_output["average"].append(mean)
                iou_output["average"].append(mean)
                # Collect classes specific IoU.
                global_iou_output[classname].append(mean)
                iou_output[classname].append(mean)
            for classname, metrics in floor["metrics"].items():
                for threshold, values in metrics["thresholds"].items():
                    for name, value in values.items():
                        if name == "matched":
                            continue
                        # Collect average over classes metrics
                        global_metrics_output[name][threshold]["average"].append(value)
                        metrics_output[name][threshold]["average"].append(value)
                        # Collect classes specific metrics
                        global_metrics_output[name][threshold][classname].append(value)
                        metrics_output[name][threshold][classname].append(value)

        data = {}
        for name, thresholds in metrics_output.items():
            data[name] = {}
            for threshold, classnames in thresholds.items():
                data[name][threshold] = {}
                for classname, values in classnames.items():
                    data[name][threshold][classname] = np.mean(values)
        data["iou"] = {}
        for classname, values in iou_output.items():
             data["iou"][classname] = np.mean(values)
        data["floors"] = floors

        logging.info("Matching data export...")
        with open(output.joinpath(f"{model}.json"), "w+", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4, cls=utils.NumpyArrayEncoder)

    with open(output.joinpath(f"scores.txt"), "w+") as file:
        file.write("submitted: 1\n")
        for name, thresholds in global_metrics_output.items():
            for threshold, classnames in thresholds.items():
                for classname, values in classnames.items():
                    file.write(f"{threshold * 100:.0f}cm_{name}_{classname}: {np.mean(values):.2f}\n")
        for classname, values in global_iou_output.items():
            file.write(f"{classname}_IoU: {np.mean(values):.2f}\n")
        file.close()
    
    with open(output.joinpath(f"scores.html"), "w+") as file:
        file.write("<h1>Detailed Submission</h1>\n")
        file.write("<h>\n")
        file.write("submitted: 1<br>\n")
        for name, thresholds in global_metrics_output.items():
            for threshold, classnames in thresholds.items():
                for classname, values in classnames.items():
                    file.write(f"{threshold * 100:.0f}cm_{name}_{classname}: {np.mean(values):.2f}<br>\n")
        for classname, values in global_iou_output.items():
            file.write(f"{classname}_IoU: {np.mean(values):.2f}<br>\n")
        file.write("</h>")
        file.close()

if __name__ == "__main__":
    app()
