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
            # Endpoint to get filter options
            if parsed_path.path == '/api/filters':
                cuisine_cursor = conn.execute("SELECT DISTINCT cuisine FROM recipes WHERE cuisine != 'Unknown' ORDER BY cuisine")
                cuisines = [row['cuisine'] for row in cuisine_cursor.fetchall()]
                
                dish_type_cursor = conn.execute("SELECT DISTINCT dish_type FROM recipes WHERE dish_type IS NOT NULL ORDER BY dish_type")
                dish_types = [row['dish_type'] for row in dish_type_cursor.fetchall()]
                
                data = {'cuisines': cuisines, 'dish_types': dish_types}

            # Endpoint to get recipes (with multiple filter options)
            elif parsed_path.path == '/api/recipes':
                query = query_params.get('q', [None])[0]
                cuisine = query_params.get('cuisine', [None])[0]
                dish_type = query_params.get('dish_type', [None])[0]
                difficulty = query_params.get('difficulty', [None])[0]

                sql_query = "SELECT id, title, image_url, cuisine, difficulty, dish_type FROM recipes"
                conditions = []
                params = []

                if query:
                    conditions.append("title LIKE ?")
                    params.append(f"%{query}%")
                if cuisine:
                    conditions.append("cuisine = ?")
                    params.append(cuisine)
                if dish_type:
                    conditions.append("dish_type = ?")
                    params.append(dish_type)
                if difficulty:
                    conditions.append("difficulty = ?")
                    params.append(difficulty)
                
                if conditions:
                    sql_query += " WHERE " + " AND ".join(conditions)
                
                sql_query += " ORDER BY image_score DESC"
                
                cursor = conn.execute(sql_query, tuple(params))
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