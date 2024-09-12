
from tree_sitter_languages import get_language, get_parser
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
from tqdm import tqdm
import os, json, gzip, argparse

def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='optional arguments')
    parser.add_argument('--output_dir', '-o',
                    metavar='PATH',
                    dest='output_dir',
                    required=False,
                    default='output',
                    help='Path to the directory where to save the cleaned files')
    
    required = parser.add_argument_group('required arguments')
    required.add_argument('--input_dir', '-i',
                        metavar='PATH',
                        dest='directory',
                        required=True,
                        help='Directory containing the files to clean')
    required.add_argument('--language', '-l',
                        metavar='NAME',
                        dest='language',
                        required=True,
                        choices=['java', 'py'],
                        help='Language of the files to clean. Possible values: java, py')
    
    return parser

def retrieve_files(dir, results_dir):
    filepaths = list()
    target_dir = os.path.join(results_dir, dir)
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".gz") and not file.endswith("results.json.gz"):
                filepaths.append(os.path.join(root, file))
    return filepaths

def copy_files(filepaths, target_dir):
    clean_filepaths = list()
    for file in filepaths:
        filepath = os.path.join(target_dir, os.path.basename(file))
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(file, 'rb') as of:
            with open(filepath, 'wb') as tf:
                tf.write(of.read())
        clean_filepaths.append(filepath)
    return clean_filepaths


def extract_function(code: str, lang: str):
    language = get_language(lang)
    parser = get_parser(lang)
    tree = parser.parse(code.encode())
    node = tree.root_node

    if lang == 'java':
        query ="""(method_declaration) @target.function"""
    elif lang == 'python':
        query ="""(function_definition) @target.function"""
      
    func_query = language.query(query)

    captures = func_query.captures(node)
    captures = captures[0][0] if captures else None
    return captures

def remove_comments_strings(code: str, language: str) -> str:
    lexer = get_lexer_by_name(language)
    tokens = list(lexer.get_tokens(code))
    # If token is a docstring, then retrieve only the newlines
    new_tokens = list()
    for token in tokens:
        if token[0] == Token.Literal.String.Doc or token[0] in Token.Comment or token[0] == Token.Literal.String.Char:
            new_tokens.append((Token, '\n' * token[1].count('\n')))
        else:
            new_tokens.append(token)      
    clean_code = ''.join([token[1] for token in new_tokens])
    return clean_code.strip()

def separate_doc_func(function, lexer, language, target_token=Token.Comment): 
    tokens = list(lexer.get_tokens(function))
    doc_tokens, function_tokens = list(), list()

    # Check the first token starting with a target_token
    if tokens[0][0] not in target_token:
        print("Found a function without a docstring.")
        print("Language:", language)
        print("Function:", function)
        print()
        return '', function
    is_start = True
    for t in tokens:
        if is_start and (t[0] in target_token or t[0] in Token.Text):
            doc_tokens.append(t[1])
        else:
            is_start = False
            function_tokens.append(t[1])
    doc = ''.join(doc_tokens)
    func = ''.join(function_tokens)
    return doc, func

if __name__ == '__main__':  

    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()

    dirs = os.listdir(args.directory)
    language_dict = {'java': 'java', 'py': 'python'}
    for dir in dirs:
        language = args.language
        print(f"Processing {dir} - Language: {language}")
        
        # Copy the same files in the output directory
        filepaths = retrieve_files(dir, args.directory)
        target_dir = os.path.join(args.output_dir, dir)
        clean_filepaths = copy_files(filepaths, target_dir)

        language = language_dict[language] if language in language_dict else None
        if not language:
            print(f"Language not supported: {language}")
            continue

        # Update the content of the files
        lexer = get_lexer_by_name(language)
        for filepath in tqdm(clean_filepaths, desc="Cleaning completions"):
            with gzip.open(filepath, 'rb') as f:
                data = f.read()
                data = json.loads(data)
                prompt = data['prompt']
                completions = data['completions']

                clean_completions = list()
                for completion in completions:
                    code = prompt + completion
                    
                    function_node = extract_function(code, language)
                    if not function_node:
                        clean_completions.append(completion)
                        continue
                    
                    start_node, end_node = function_node.start_point, function_node.end_point
                    start_line, start_column, end_line, end_column = start_node[0], start_node[1], end_node[0], end_node[1]
                    
                    code_lines = code.split("\n")
                    code_lines = code_lines[:end_line+1]
                    code_lines[-1] = code_lines[-1][:end_column]
                    code = "\n".join(code_lines)

                    # Remove initial prompt
                    code = code[len(prompt):]
                    clean_completions.append(code)
                
                data['completions'] = clean_completions
                with gzip.open(filepath, 'wb') as f:
                    f.write(json.dumps(data).encode())
