import os

def count_lines_in_file(file_path):
    """Count the number of lines in a given file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        return sum(1 for line in file)

def count_lines_in_project(root_dir):
    """Count the total number of lines in all .py and .ipynb files in the project."""
    total_lines = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py') or filename.endswith('.ipynb'):
                file_path = os.path.join(dirpath, filename)
                total_lines += count_lines_in_file(file_path)
    return total_lines

if __name__ == "__main__":
    project_root = os.getcwd()
    total_lines = count_lines_in_project(project_root)
    print(f"Total number of lines in .py and .ipynb files: {total_lines}")
