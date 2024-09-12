import os, gzip, json, argparse

def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='optional arguments')
    parser.add_argument('--output_filepath', '-o',
                        metavar='FILEPATH',
                        dest='output_filepath',
                        required=False,
                        type=str,
                        default='output.csv',
                        help='Name of the file to save the results')

    required = parser.add_argument_group('required arguments')
    required.add_argument('--folder_path', '-i',
                        metavar='PATH',
                        dest='folder_path',
                        required=True,
                        type=str,
                        help='Path to the folder containing the RQ results')
    
    return parser


def get_model_quantization_language(path):
    """
    Extract model, quantization, and language from the path name.

    :param path: Path from which to extract model, quantization, and language.
    :return: Tuple containing model, quantization, and language information.
    """
    print(f"Extracting model, quantization, and language from path: {path}")
    model_name = path.split("/")[1]
    model_name, quantization = model_name.split("b-")[0] + 'b', model_name.split("b-")[1]
    model_name = model_name.replace('converted-', '')

    language = path.split("/")[2]
    language = language.split("_")[0]
    
    return model_name, quantization, language


def main():
    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()

    folder_path = args.folder_path
    output = args.output_filepath

    # Create CSV and add header
    csv_file = open(output, "w")
    csv_file.write("model,quantization,language,problem,pass\n")

    # Recursively traverse folder:
    for root, dirs, files in os.walk(folder_path):
        print(f"Processing folder: {root}")
        model, quantization, language = None, None, None
        # If folder contains results gz files, process them and update CSV
        for file in files:
            if file.endswith(".results.json.gz"):
                if model is None:
                    model, quantization, language = get_model_quantization_language(root)
                with gzip.open(os.path.join(root, file), 'rb') as f:
                    # Read content as JSON
                    problem_results = json.loads(f.read())
                    # Add results to CSV
                    for result in problem_results["results"]:
                        csv_file.write(f"""{model},{quantization},{language},{problem_results["name"]},{1 if result["status"] == "OK" and result["exit_code"] == 0 else 0}\n""")


if __name__ == "__main__":
    main()