import http.server
import socketserver
import json
import sqlite3
from urllib.parse import urlparse, parse_qs

PORT = 8000
DB_NAME = 'recipes.db'

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

class RecipeServer(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # API endpoints
        if self.path.startswith('/api/'):
            self.handle_api_request()
        # Serve index.html for the root path
        elif self.path == '/':
            self.path = '/viewRecipeDb.html'
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
        # Serve other static files
        else:
            return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def handle_api_request(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)
        
        conn = get_db_connection()
        
        data = []

        try:
            # Endpoint to get all unique batches
            if parsed_path.path == '/api/batches':
                batches_cursor = conn.execute('SELECT DISTINCT batch_id FROM recipes ORDER BY batch_id')
                batches_rows = batches_cursor.fetchall()
                
                for row in batches_rows:
                    batch_id = row['batch_id']
                    # Get a few ingredients to represent the batch
                    ingredients_cursor = conn.execute(
                        'SELECT ingredients_json FROM recipes WHERE batch_id = ? LIMIT 1', (batch_id,)
                    )
                    ingredients_row = ingredients_cursor.fetchone()
                    if ingredients_row:
                        ingredients = json.loads(ingredients_row['ingredients_json'])
                        data.append({'id': batch_id, 'ingredients': ingredients})

            # Endpoint to get recipes (filtered by query or batch)
            elif parsed_path.path == '/api/recipes':
                query = query_params.get('q', [None])[0]
                batch_id = query_params.get('batch_id', [None])[0]
                
                if query:
                    cursor = conn.execute("SELECT id, title, image_url FROM recipes WHERE title LIKE ?", ('%' + query + '%',))
                elif batch_id:
                    cursor = conn.execute("SELECT id, title, image_url FROM recipes WHERE batch_id = ?", (batch_id,))
                else:
                    cursor = conn.execute("SELECT id, title, image_url FROM recipes")
                
                rows = cursor.fetchall()
                data = [dict(row) for row in rows]
            
            # Endpoint to get a single recipe by ID
            elif parsed_path.path.startswith('/api/recipe/'):
                recipe_id = parsed_path.path.split('/')[-1]
                cursor = conn.execute("SELECT * FROM recipes WHERE id = ?", (recipe_id,))
                row = cursor.fetchone()
                if row:
                    recipe_data = dict(row)
                    # The JSON fields need to be parsed
                    recipe_data['ingredients'] = json.loads(recipe_data['ingredients_json'])
                    recipe_data['instructions'] = json.loads(recipe_data['instructions_json'])
                    data = recipe_data

        except Exception as e:
            print(f"Database error: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Internal server error'}).encode('utf-8'))
            return
        
        finally:
            conn.close()
            
        self.wfile.write(json.dumps(data).encode('utf-8'))

with socketserver.TCPServer(("", PORT), RecipeServer) as httpd:
    print(f"Serving at http://localhost:{PORT}")
    httpd.serve_forever() 