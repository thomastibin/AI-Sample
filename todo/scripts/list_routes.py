from pprint import pprint
import importlib
import sys
import pathlib
# Ensure repo root is on sys.path
repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

m = importlib.import_module('mcpserver.server')
app = m.app
routes = []
for r in app.routes:
    try:
        path = getattr(r, 'path', None) or getattr(r, 'prefix', None) or str(r)
        methods = getattr(r, 'methods', None)
        routes.append((path, type(r).__name__, methods))
    except Exception as e:
        routes.append((str(r), type(r).__name__, 'ERR:'+repr(e)))

pprint(routes)
