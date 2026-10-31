import ast
import base64
import io
import json
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def test_cells_compile_and_embedded_source_matches_repository(self):
        notebook = json.loads((ROOT / 'colab/Finetune_MOESCORE_Expert4.ipynb').read_text())
        self.assertEqual(notebook['nbformat'], 4)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')
        source = ''.join(notebook['cells'][5]['source'])
        assignment = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'BUNDLE' for target in node.targets))
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(assignment.value)))) as archive:
            self.assertIn('scripts/predict_expert4.py', archive.namelist())
            for name in archive.namelist():
                self.assertEqual(archive.read(name), (ROOT / name).read_bytes(), name)

    def test_colab_dependency_cell_rejects_python_313_and_uses_supported_numpy(self):
        notebook = json.loads((ROOT / 'colab/Finetune_MOESCORE_Expert4.ipynb').read_text())
        runtime = ''.join(notebook['cells'][1]['source'])
        dependencies = ''.join(notebook['cells'][2]['source'])
        self.assertIn('sys.version_info >= (3, 13)', runtime)
        self.assertIn('numpy==1.26.4', dependencies)
        self.assertIn('laion-clap==1.1.7', dependencies)
        self.assertIn('--only-binary=numpy', dependencies)
        self.assertNotIn('laion-clap==1.1.6', dependencies)

    def test_dependency_setup_uses_fresh_python_processes(self):
        notebook = json.loads((ROOT / 'colab/Finetune_MOESCORE_Expert4.ipynb').read_text())
        for index in (1, 2):
            source = ''.join(notebook['cells'][index]['source'])
            imports = [alias.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Import)
                       for alias in node.names]
            self.assertFalse({'torch', 'torchaudio', 'numpy', 'laion_clap'} & set(imports), source)
        dependencies = ''.join(notebook['cells'][2]['source'])
        self.assertIn('subprocess.check_call', dependencies)
        self.assertIn('torchaudio, laion_clap, transformers', dependencies)
        self.assertIn("[sys.executable, '-c', dependency_probe]", dependencies)


if __name__ == '__main__':
    unittest.main()
